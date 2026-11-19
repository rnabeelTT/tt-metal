# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Multi-user generation engine for Hy4-flash: up to model.slots sequences, one batched decode pass per step.

Each sequence owns a model slot (its cache regions, tt/model.py set_slot) and, while it decodes, a lane of the batched
decoder (tt/decode_batch.py). Prefill runs per sequence on the single path into its slot; decode steps every live
sequence in one pass (one row each). Sampling (temperature / top-k / top-p / seed) runs on the host per sequence over
its fp32 logits (tt/engine.py HostSampler). Solo (HY4_BATCH_SOLO=1, default): while a step has one sequence, it leaves
its lane (Hy4BatchDecoder.remove: windows / group keys back to its slot) and decodes on the single path (no lane
overhead); the next multi-sequence step or prefill puts it back (add). No speculative decoding here.

Contract (tt/generator_vllm.py): alloc() -> a free slot; prefill(slot, ids, sampling) -> first token;
step({slot: (tok, pos, authoritative)}) -> {slot: [next]}; step_logits({slot: (tok, pos)}) -> {slot: logits};
finish(slot) frees it.
"""

from __future__ import annotations

import os
import time

import torch
from loguru import logger

from models.autoports.tencent_hy4_flash.tt.engine import HostSampler


class Hy4BatchEngine:
    multi = True

    def __init__(self, model, trace: bool = True):
        from models.autoports.tencent_hy4_flash.tt.decode_batch import Hy4BatchDecoder

        self.model, self.cfg = model, model.cfg
        self.S = int(model.slots)
        lanes = next(b for b in (2, 4, 8, 16, 32) if b >= self.S)
        self.dec = Hy4BatchDecoder(model, lanes=lanes, trace=trace)
        self.trace = bool(trace)
        self.V = int(self.cfg.vocab)
        self.eos = {int(self.cfg.eos)}
        self.chunk = int(model.chunk)
        self.seqs = {}  # slot -> {"lane", "tok", "pos", "smp", "adopt", "toks"}
        self.solo = None  # the slot decoding on the single path (its lane released)
        self.solo_on = os.environ.get("HY4_BATCH_SOLO", "1") == "1"
        self.last_logits = {}
        self.stats = {"steps": 0, "rows": 0, "t_step": 0.0, "t_prefill": 0.0, "prefills": 0, "prefill_tokens": 0}

    # ------------------------------------------------------------------ slots
    def alloc(self) -> int:
        free = [s for s in range(self.S) if s not in self.seqs]
        if not free:
            raise RuntimeError(f"Hy4: all {self.S} slots busy")
        return free[0]

    def finish(self, slot: int):
        q = self.seqs.pop(int(slot), None)
        if self.solo == int(slot):
            self.solo = None
        elif q is not None:
            self.dec.drop(q["lane"])
        self.last_logits.pop(int(slot), None)

    @property
    def live(self):
        return sorted(self.seqs)

    # ------------------------------------------------------------------ prefill / decode
    def prefill(self, slot: int, prompt_ids, sampling=None) -> int:
        t0 = time.time()
        slot = int(slot)
        self.finish(slot)
        if self.solo is not None:  # the attention staging windows are shared: the solo sequence back to a lane first
            self._exit_solo()
        ids = torch.as_tensor(prompt_ids, dtype=torch.long).reshape(-1)
        n = int(ids.numel())
        if n + 64 > self.model.max_ctx:
            raise ValueError(f"Hy4: prompt of {n} tokens exceeds the slot context {self.model.max_ctx - 64}")
        m = self.model
        m.set_slot(slot)
        m.reset()
        lg = m.prefill(ids)
        lane = self.dec.add(slot, n, ids.tolist())
        smp = HostSampler(sampling)
        tok = smp(lg)
        self.seqs[slot] = {"lane": lane, "tok": tok, "pos": n, "smp": smp, "adopt": False, "toks": ids.tolist()}
        self.last_logits[slot] = lg
        dt = time.time() - t0
        self.stats["t_prefill"] += dt
        self.stats["prefills"] += 1
        self.stats["prefill_tokens"] += n
        logger.info(f"Hy4 prefill {n} tokens into slot {slot} (lane {lane}) in {dt:.2f} s; live {self.live}")
        return tok

    def _enter_solo(self, slot):
        q = self.seqs[slot]
        ln = self.dec.lanes[q["lane"]]
        self.dec.remove(q["lane"])  # model slot = slot, its pos / Engram state; staging windows = the lane's
        d = self.dec
        d.ob, d.ogb, d.pos = ln["ob"], ln["ogb"], ln["pos"]
        d.set_history(q["toks"])
        q["lane"] = None
        self.solo = slot

    def _exit_solo(self):
        slot, self.solo = self.solo, None
        q = self.seqs[slot]
        q["lane"] = self.dec.add(slot, q["pos"], q["toks"])

    def _pass(self, rows: dict) -> dict:
        """rows {slot: [ids]} -> {slot: logits [n, V]} (solo: the single path)."""
        t0 = time.time()
        if self.solo_on and len(rows) == 1 and self.S > 1:
            (slot, ids), = rows.items()
            if self.solo not in (None, slot):
                self._exit_solo()
            if self.solo is None:
                self._enter_solo(slot)
            out = {slot: self.dec.run(list(ids), logits_rows="all", p0=self.seqs[slot]["pos"]).reshape(len(ids), -1)}
        else:
            if self.solo is not None:
                self._exit_solo()
            lo = self.dec.run_batch({self.seqs[s]["lane"]: ids for s, ids in rows.items()})
            out = {s: lo[self.seqs[s]["lane"]] for s in rows}
        for s, ids in rows.items():
            self.seqs[s]["toks"] += [int(t) for t in ids]
        self.stats["t_step"] += time.time() - t0
        self.stats["steps"] += 1
        self.stats["rows"] += len(rows)
        return out

    def _check_pos(self, slot, q, pos):
        if self.model.max_ctx - 64 <= q["pos"]:
            raise RuntimeError(f"Hy4 slot {slot}: context full ({q['pos']} tokens)")
        if pos is not None and int(pos) != q["pos"]:
            raise RuntimeError(f"Hy4 slot {slot}: step at pos {pos}, engine at {q['pos']}")

    def step(self, reqs: dict) -> dict:
        """reqs {slot: (tok, pos, host_authoritative)} -> {slot: [next token]} (one batched pass)."""
        rows = {}
        for slot, (tok, pos, auth) in reqs.items():
            q = self.seqs[slot]
            adopt, q["adopt"] = q["adopt"], False
            if tok is not None and (q["tok"] is None or ((auth or adopt) and int(tok) != q["tok"])):
                self._check_pos(slot, q, pos)
                q["tok"] = int(tok)
            self._check_pos(slot, q, None)
            rows[slot] = [q["tok"]]
        out = self._pass(rows)
        res = {}
        for slot, lg in out.items():
            q = self.seqs[slot]
            nxt = q["smp"](lg[0])
            q["tok"], q["pos"] = nxt, q["pos"] + 1
            res[slot] = [nxt]
        return res

    def step_logits(self, reqs: dict) -> dict:
        """Host-sampled step: reqs {slot: (tok, pos)} (each continues its sequence) -> {slot: next-token logits}."""
        rows = {}
        for slot, (tok, pos) in reqs.items():
            q = self.seqs[slot]
            self._check_pos(slot, q, pos)
            rows[slot] = [int(tok)]
        out = self._pass(rows)
        res = {}
        for slot, lg in out.items():
            q = self.seqs[slot]
            q["pos"] += 1
            q["tok"], q["adopt"] = None, True
            res[slot] = lg[0]
        return res

    # ------------------------------------------------------------------ warm-up
    def warm(self, prompt_len=None):
        """Compile every program serving runs before any trace is captured: the prefill buckets (slot 0), then each
        batch size 1..S at every score width bucket (eager), then capture the common traces."""
        t0 = time.time()
        g = torch.Generator().manual_seed(0)
        rnd = lambda n: torch.randint(1000, 100000, (n,), generator=g).tolist()  # noqa: E731
        m = self.model
        if os.environ.get("HY4_PF_BUCKETS", "1") == "1":
            for b in m.PF_BUCKETS:
                if b <= self.chunk and b < m.max_ctx:
                    self.prefill(0, rnd(b - 1))
                    m.warm_head_blocks(b)
            # later chunks at every start the slot context allows (their programs change with the indexer's score width
            # bucket; a 12K prompt after a 6K warm compiled 24 programs at serve time and released every trace)
            m_fit = max(0, (m.max_ctx - 64 - self.chunk) // self.chunk)
            for k in range(1, int(os.environ.get("HY4_WARM_LONG", str(m_fit))) + 1):
                for b in m.PF_BUCKETS:
                    L = self.chunk * k + b - 1
                    if b <= self.chunk and L + 64 < m.max_ctx:
                        self.prefill(0, rnd(L))
                        m.warm_head_blocks(b)
        logger.info(f"warm: prefill buckets done; {m.dram_report()}")
        # batched programs: slot 0 past the dense regime (its rows select), the others short (dense rows)
        long_n = self.chunk + 1000 if self.chunk + 1064 < m.max_ctx else int(prompt_len or 64)
        for s in range(self.S):
            self.prefill(s, rnd(long_n if s == 0 else int(prompt_len or 64)))
        dec = self.dec
        top = dec.L * dec.gcap
        widths = sorted({w for w in (2048 * 2**i for i in range(12)) if w + 32 <= top + 32} | {top})
        dec.trace = False
        # R = 1 runs solo (the single path) when HY4_BATCH_SOLO=1: slot 0 selects, the others are dense; every slot once,
        # so every lane is unloaded once (indexer.lane_unload slices at lane * gcap: one program per lane; a lane first
        # unloaded at serve time compiled and released every trace, served run 2026-11-19)
        solo = [{s: (None, None, False)} for s in range(self.S)] if self.solo_on and self.S > 1 else []
        for R in range(1, self.S + 1):
            for w in widths:
                dec.min_width = w
                self.step({s: (None, None, False) for s in range(R)})
        for r in solo:
            self.step(r)
        dec.min_width = 0
        # dense-only passes (no row selects: no scoring) at every batch size: slot 0 short again
        self.prefill(0, rnd(int(prompt_len or 64)))
        for R in range(2, self.S + 1):
            self.step({s: (None, None, False) for s in range(R)})
        dec.trace = self.trace
        logger.info(f"warm: eager batched passes done; {m.dram_report()}")
        steps = int(os.environ.get("HY4_WARM_STEPS", "3"))
        for R in range(1, self.S + 1):
            for _ in range(steps):
                self.step({s: (None, None, False) for s in range(R)})
        for r in solo[:2]:  # capture the single-path traces of both regimes
            for _ in range(steps):
                self.step(r)
        for s in range(self.S):
            self.finish(s)
            m.set_slot(s)
            m.reset()
        m.set_slot(0)
        logger.info(f"warm: traces captured; {m.dram_report()}")
        self.stats = {k: 0 if isinstance(v, int) else 0.0 for k, v in self.stats.items()}
        logger.info(f"Hy4 batch engine warm-up {time.time() - t0:.1f} s ({self.S} slots, widths {widths}, "
                    f"{len(dec.btraces)} batched traces, {len(dec.traces)} single)")

    def capture(self):
        """Batched traces are captured on first use per (rows, width) by tt/decode_batch.Hy4BatchDecoder."""
