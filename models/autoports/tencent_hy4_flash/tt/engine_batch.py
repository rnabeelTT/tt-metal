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

import numpy as np
import torch
from loguru import logger

from models.autoports.tencent_hy4_flash.tt.engine import HostSampler


class Hy4BatchEngine:
    multi = True

    def __init__(self, model, trace: bool = True, k: int = 0):
        from models.autoports.tencent_hy4_flash.tt.decode_batch import Hy4BatchDecoder, Hy4BatchSpecDecoder

        self.model, self.cfg = model, model.cfg
        self.S = int(model.slots)
        lanes = next(b for b in (2, 4, 8, 16, 32) if b >= self.S)
        # k > 0: DFly2 speculative decoding while one sequence decodes alone (the solo single path)
        self.k = int(k) if os.environ.get("HY4_BATCH_SOLO", "1") == "1" else 0
        if self.k > 0:
            self.dec = Hy4BatchSpecDecoder(model, lanes=lanes, trace=trace, k=self.k)
            self.k = self.dec.k
        else:
            self.dec = Hy4BatchDecoder(model, lanes=lanes, trace=trace)
        self.trace = bool(trace)
        self.V = int(self.cfg.vocab)
        self.eos = {int(self.cfg.eos)}
        self.chunk = int(model.chunk)
        self.seqs = {}  # slot -> {"lane", "tok", "pos", "smp", "adopt", "toks"}
        self.solo = None  # the slot decoding on the single path (its lane released)
        self.solo_on = os.environ.get("HY4_BATCH_SOLO", "1") == "1"
        self.last_logits = {}
        self.stats = {"steps": 0, "rows": 0, "t_step": 0.0, "t_prefill": 0.0, "prefills": 0, "prefill_tokens": 0,
                      "reused_tokens": 0}
        # prefix reuse (HY4_PREFIX_CACHE, default 1): a finished sequence's slot keeps its caches and token history; a new
        # prompt sharing a prefix with it continues from there (tt/model.py rewind) instead of a full prefill
        self.pc_on = os.environ.get("HY4_PREFIX_CACHE", "1") == "1"
        self.pc_min = int(os.environ.get("HY4_PC_MIN", "512"))
        # HY4_PC_EXACT (default 1): reuse only positions the kept sequence's prefill computed in whole chunks at a fresh
        # prefill's chunk boundaries (0, 2048, 2048 + chunk, ...), resuming at such a boundary: every chunk then runs
        # exactly as in a fresh prefill of the new prompt (same rows, same programs): bit-identical output. 0: any
        # 512-aligned prefix (dense regime too, decoded positions too): device-noise-level differences
        self.pc_exact = os.environ.get("HY4_PC_EXACT", "1") == "1"
        # the single path's indexer state in fixed buffers (_solo_in): its traces are keyed (rows, regime, bucket), not
        # by slot, and capture the addresses of the active slot's group keys / staging windows (per-slot tensors): a
        # sequence decoding solo in another slot than the capture's read and wrote the capture slot's group keys
        import ttnn

        self._sb = []
        for layer in model.layers:
            ix = layer.attn.indexer
            if ix is not None:
                z = lambda n: ttnn.zeros((1, 1, n, ix.d), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,  # noqa: E731
                                         device=model.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                self._sb.append((ix, (z(ix.gcap), z(64), z(64))))
        self.kept = {}  # slot -> {"toks": the finished sequence's fed tokens (positions 0 ..), "t": finish time, "img",
        #                          "full": its positions < full were computed by whole prefill chunks (_full)}

    # ------------------------------------------------------------------ slots
    def alloc(self, taken=(), ids=None) -> int:
        """A free slot: (ids given, prefix reuse on) the one whose kept sequence shares the longest reusable prefix with
        ids (ties: the lowest), else the least recently finished (never used first)."""
        free = [s for s in range(self.S) if s not in self.seqs and s not in taken]
        if not free:
            raise RuntimeError(f"Hy4: all {self.S} slots busy")
        if not self.pc_on:
            return free[0]
        if ids is not None:
            ids = [int(t) for t in ids]
            r, s = max((self._reuse(s, ids), -s) for s in free)
            if r > 0:
                return -s
        return min(free, key=lambda s: (self.kept.get(s, {}).get("t", 0.0), s))

    def finish(self, slot: int):
        q = self.seqs.pop(int(slot), None)
        if self.solo == int(slot):
            self.solo = None
            self._solo_out(int(slot))
        elif q is not None:
            if self.pc_on:
                self._park(q["lane"])
            else:
                self.dec.drop(q["lane"])
        if q is not None and self.pc_on:
            self.kept[int(slot)] = {"toks": q["toks"], "t": time.time(), "img": bool(q.get("img")),
                                    "full": int(q.get("full", 0))}
        self.last_logits.pop(int(slot), None)

    # ------------------------------------------------------------------ prefix reuse
    def _park(self, lane):
        """A finished sequence's lane freed with its indexers' group keys written back to its slot (decode in a lane
        writes them only to the lane's region; the KV rows go to the slot's cache directly)."""
        d = self.dec
        ln = d.lanes[lane]
        m = self.model
        m.set_slot(ln["slot"])
        _, _, Ms = d._onehots(lane)
        for layer in m.layers:
            if layer.attn.indexer is not None:
                layer.attn.indexer.lane_unload(d.L, lane, Ms)
        d.lanes[lane] = None

    def _big(self) -> int:
        """The prefill chunk past the dense regime (tt/model.py prefill)."""
        return min(int(os.environ.get("HY4_PREFILL_CHUNK", self.model.chunk)), self.model.chunk)

    def _full(self, n) -> int:
        """Positions < this of an n-token prompt are computed by whole chunks of a fresh prefill (boundaries 2048,
        2048 + chunk, ...; the last, partial chunk is padded to a bucket and differs with the prompt length)."""
        dense = int(self.cfg.idx_candidate_topk)
        if n < dense or (n == dense and os.environ.get("HY4_PF_TRACE", "0") == "1"):  # (a traced first chunk)
            return 0
        return dense + ((n - dense) // self._big()) * self._big()

    def _reuse(self, slot, ids) -> int:
        """Positions of `ids` (list) the free slot's kept sequence can serve: r <= the common prefix d, r < len(ids) (the
        last row runs for its logits). HY4_PC_EXACT=1: r a fresh prefill's chunk boundary <= the kept prefill's whole
        chunks (bit-identical). Else aligned for the continued prefill (r < 2048, the dense regime: a multiple of
        HY4_PC_ALIGN_DENSE, default 512 >= the SWA window, so the chunk takes the flash SWA path; else of
        HY4_PC_ALIGN, default 512: the group-key slab writes of a chunk at group offset r / 4 then take the slab sizes
        warm() compiles), the SWA ring
        still holding positions [r - window, r), r >= HY4_PC_MIN. 0: no reuse."""
        k = self.kept.get(int(slot))
        m = self.model
        if (not self.pc_on or k is None or k["img"] or int(slot) in self.seqs
                or (m.engram is not None and not hasattr(m.engram, "hist"))):
            return 0
        toks = k["toks"]
        lim = min(len(toks), len(ids) - 1)
        if lim < self.pc_min:
            return 0
        a = np.asarray(toks[:lim], dtype=np.int64)
        b = np.asarray(ids[:lim], dtype=np.int64)
        neq = np.flatnonzero(a != b)
        dpre = int(neq[0]) if neq.size else lim
        dense = int(self.cfg.idx_candidate_topk)
        if self.pc_exact:  # a chunk boundary of a fresh prefill, inside the kept prefill's whole chunks
            c = min(dpre, int(k.get("full", 0)))
            r = dense + ((c - dense) // self._big()) * self._big() if c >= dense else 0
        elif dpre >= dense:
            A = int(os.environ.get("HY4_PC_ALIGN", "512"))
            r = (dpre // A) * A
            if r < dense:
                r = dense
        else:
            A = int(os.environ.get("HY4_PC_ALIGN_DENSE", "512"))
            r = (dpre // A) * A
        if r <= 0 or r % 128 or r < max(self.pc_min, int(self.cfg.window)):
            return 0
        swa = next((l.attn for l in m.layers if l.attn.kind == "swa"), None)
        if swa is not None:  # ring rows hold the last swa.rows positions written (prefill padding, decode lookahead)
            hiw = max(getattr(m, "_hiw", {}).get(int(slot), 0), len(toks) + 96 + self.k)
            if r - int(self.cfg.window) < hiw - swa.rows:
                return 0
        return r

    @property
    def live(self):
        return sorted(self.seqs)

    # ------------------------------------------------------------------ prefill / decode
    def prefill(self, slot: int, prompt_ids, sampling=None, mm=None) -> int:
        """mm: image inputs of the prompt (Hy4Model.set_images kwargs, tt/vision.py Hy4Vision.prepare)."""
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
        idl = ids.tolist()
        r = self._reuse(slot, idl) if mm is None else 0
        m.set_slot(slot)
        if r:
            m.rewind(r, idl)  # the slot's kept sequence serves positions < r
        else:
            m.reset()
        self.kept.pop(slot, None)
        if mm is not None:
            m.set_images(**mm)
        lg = m.prefill(ids[r:])
        m.prefill_caches_release()  # (before any decode replay: tt/model.py)
        lane = self.dec.add(slot, n, idl)
        smp = HostSampler(sampling)
        tok = smp(lg)
        self.seqs[slot] = {"lane": lane, "tok": tok, "pos": n, "smp": smp, "adopt": False, "toks": idl,
                           "img": mm is not None, "full": self._full(n)}
        self.last_logits[slot] = lg
        dt = time.time() - t0
        self.stats["t_prefill"] += dt
        self.stats["prefills"] += 1
        self.stats["prefill_tokens"] += n - r
        self.stats["reused_tokens"] += r
        logger.info(f"Hy4 prefill {n} tokens ({r} reused) into slot {slot} (lane {lane}) in {dt:.2f} s; live {self.live}")
        return tok

    def prefill_many(self, reqs) -> list:
        """reqs [(slot, ids, sampling)] -> first tokens. Simultaneous short prompts (dense regime, ordinary text) run as
        one packed prefill pass (Hy4Model.forward_packed: B prompts on the attention's batch dimension); otherwise one
        prefill each."""
        m = self.model
        idl = [torch.as_tensor(ids, dtype=torch.long).reshape(-1) for _, ids, _ in reqs]
        if self.pc_on and len(reqs) > 1:  # prompts continuing a kept sequence: their own (continued) prefill
            for slot, _, _ in reqs:
                self.finish(slot)
            hit = [self._reuse(slot, t.tolist()) > 0 for (slot, _, _), t in zip(reqs, idl)]
            if any(hit):
                out = [None] * len(reqs)
                rest = [i for i, h in enumerate(hit) if not h]
                for i, tok in zip(rest, self.prefill_many([reqs[i] for i in rest]) if rest else []):
                    out[i] = tok
                for i in [i for i, h in enumerate(hit) if h]:
                    out[i] = self.prefill(reqs[i][0], idl[i], reqs[i][2])
                return out
        plan = m.pack_plan([int(t.numel()) for t in idl])
        if plan is None or not all(m.engram_keep_all(t) for t in idl):
            return [self.prefill(slot, ids, sp) for (slot, _, sp), ids in zip(reqs, idl)]
        t0 = time.time()
        for slot, _, _ in reqs:
            self.finish(slot)
        if self.solo is not None:
            self._exit_solo()
        for slot, _, _ in reqs:
            self.kept.pop(slot, None)
        lgs = m.forward_packed([(slot, ids) for (slot, _, _), ids in zip(reqs, idl)], plan)
        m.prefill_caches_release()
        out = []
        for (slot, _, sp), ids, lg in zip(reqs, idl, lgs):
            lane = self.dec.add(slot, int(ids.numel()), ids.tolist())
            smp = HostSampler(sp)
            tok = smp(lg)
            self.seqs[slot] = {"lane": lane, "tok": tok, "pos": int(ids.numel()), "smp": smp, "adopt": False,
                               "toks": ids.tolist()}
            self.last_logits[slot] = lg
            out.append(tok)
        dt = time.time() - t0
        self.stats["t_prefill"] += dt
        self.stats["prefills"] += len(reqs)
        self.stats["prefill_tokens"] += sum(int(t.numel()) for t in idl)
        logger.info(f"Hy4 packed prefill of {len(reqs)} prompts {[int(t.numel()) for t in idl]} (plan {plan}) into slots "
                    f"{[r[0] for r in reqs]} in {dt:.2f} s; live {self.live}")
        return out

    def _enter_solo(self, slot):
        q = self.seqs[slot]
        ln = self.dec.lanes[q["lane"]]
        self.dec.remove(q["lane"])  # model slot = slot, its pos / Engram state; staging windows = the lane's
        d = self.dec
        d.ob, d.ogb, d.pos = ln["ob"], ln["ogb"], ln["pos"]
        d.set_history(q["toks"])
        q["lane"] = None
        self.solo = slot
        self._solo_in(slot)

    _SOLO_ATTRS = ("gkeys", "tk_stage", "g_stage")

    def _solo_in(self, slot):
        """The slot's indexer state (group keys, token / group windows) -> the fixed solo buffers, which the indexers then
        use as the slot's (the single-path traces read and write them)."""
        import ttnn

        self.model.set_slot(slot)
        for ix, bufs in self._sb:
            orig = tuple(getattr(ix, a) for a in self._SOLO_ATTRS)
            for src, dst in zip(orig, bufs):
                ttnn.copy(src, dst)
            ix._solo_orig = orig
            for a, b in zip(self._SOLO_ATTRS, bufs):
                setattr(ix, a, b)

    def _solo_out(self, slot):
        """The solo buffers -> the slot's own tensors (restored)."""
        import ttnn

        self.model.set_slot(slot)
        for ix, bufs in self._sb:
            orig = ix.__dict__.pop("_solo_orig")
            for src, dst in zip(bufs, orig):
                ttnn.copy(src, dst)
            for a, t in zip(self._SOLO_ATTRS, orig):
                setattr(ix, a, t)

    def _exit_solo(self):
        slot, self.solo = self.solo, None
        self._solo_out(slot)
        q = self.seqs[slot]
        d = self.dec
        q["lane"] = d.add(slot, q["pos"], q["toks"], stage=(d.ob, d.ogb))  # its current single-path windows
        q["n"] = None

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
        solo = self.solo is not None
        for s, ids in rows.items():
            self.seqs[s]["toks"] += [int(t) for t in ids]
            self.seqs[s]["n"] = 0 if solo else None  # spec: the drafter continues from a solo 1-row pass's row 0
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
        """reqs {slot: (tok, pos, host_authoritative)} -> {slot: [next token(s)]} (one batched pass; a sequence alone
        on the solo path with k > 0: one verify pass, 1 .. k + 1 tokens)."""
        rows = {}
        for slot, (tok, pos, auth) in reqs.items():
            q = self.seqs[slot]
            adopt, q["adopt"] = q["adopt"], False
            if tok is not None and (q["tok"] is None or ((auth or adopt) and int(tok) != q["tok"])):
                self._check_pos(slot, q, pos)
                q["tok"] = int(tok)
            self._check_pos(slot, q, None)
            rows[slot] = [q["tok"]]
        if self.k > 0 and len(rows) == 1 and self.solo is not None and self.solo in rows:
            q = self.seqs[self.solo]
            if q.get("n") is not None and q["pos"] + self.k + 1 + 64 <= self.model.max_ctx:
                return {self.solo: self._spec_step(self.solo)}
        out = self._pass(rows)
        res = {}
        for slot, lg in out.items():
            q = self.seqs[slot]
            nxt = q["smp"](lg[0])
            q["tok"], q["pos"] = nxt, q["pos"] + 1
            res[slot] = [nxt]
        return res

    def _spec_step(self, slot) -> list:
        """One verify pass of the solo sequence: rows [tok, d_1 .. d_k] (DFly2 drafts on its last solo pass); greedy:
        the matching prefix + the target's next token; sampled: exact speculative sampling (tt/engine.py)."""
        t0 = time.time()
        q, d, smp = self.seqs[slot], self.dec, self.seqs[slot]["smp"]
        drafts = d.draft(q["tok"], q["pos"], q["n"])[: self.k]
        rows = [q["tok"]] + drafts
        lgs = d.run(rows, logits_rows="all", p0=q["pos"])  # [k + 1, V]
        if smp.greedy:
            picks = lgs.argmax(-1).tolist()
            acc = 0
            while acc < len(drafts) and picks[acc] == drafts[acc]:
                acc += 1
            emit = drafts[:acc] + [int(picks[acc])]
        else:
            acc, emit = 0, []
            while acc < len(drafts):
                p = smp.probs(lgs[acc])
                dd = drafts[acc]
                if smp.uniform() < float(p[dd]):
                    acc += 1
                    continue
                p[dd] = 0.0
                emit = drafts[:acc] + [smp.draw(p / p.sum())]
                break
            if not emit:
                emit = drafts + [smp.draw(smp.probs(lgs[len(drafts)]))]
        q["toks"] += rows[: len(emit)]  # the accepted rows' tokens (fed at pos .. pos + acc)
        q["n"] = acc
        q["pos"] += len(emit)
        q["tok"] = emit[-1]
        self.last_logits[slot] = lgs[acc]
        self.stats["t_step"] += time.time() - t0
        self.stats["steps"] += 1
        self.stats["rows"] += 1
        self.stats["spec_emitted"] = self.stats.get("spec_emitted", 0) + len(emit)
        self.stats["spec_passes"] = self.stats.get("spec_passes", 0) + 1
        return emit

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
        m._pf_hold = True  # (tt/model.py prefill_caches_release: kept until prefill_caches_mark below)
        if os.environ.get("HY4_PF_BUCKETS", "1") == "1":
            for b in m.PF_BUCKETS:
                if b <= self.chunk and b < m.max_ctx:
                    self.prefill(0, rnd(b - 1))
                    m.warm_head_blocks(b)
            # later chunks at every start the slot context allows (their programs change with the indexer's score width
            # bucket; a 12K prompt after a 6K warm compiled 24 programs at serve time and released every trace)
            # HY4_WARM_LONG=auto (default): every (score bucket, row bucket) pair once (tt/model.py warm_long: one long
            # prefill + rewinds; before: every chunk start x row bucket from position 0, O(n^2) at long slots)
            m_fit = max(0, (m.max_ctx - 64 - self.chunk) // self.chunk)
            wl = os.environ.get("HY4_WARM_LONG", "auto")
            if wl == "auto":
                self.finish(0)
                m.warm_long(rnd, slot=0)
            for k in range(1, (0 if wl == "auto" else int(wl)) + 1):
                for b in m.PF_BUCKETS:
                    L = self.chunk * k + b - 1
                    if b <= self.chunk and L + 64 < m.max_ctx:
                        self.prefill(0, rnd(L))
                        m.warm_head_blocks(b)
        # continued prefills from a reused prefix in the dense regime (r = 512 / 1024 / 1536: a chunk to 2048 through the
        # sparse layers' listed path and the group-key slab writes at that offset): every bucket
        if self.pc_on and not self.pc_exact:
            A = int(os.environ.get("HY4_PC_ALIGN_DENSE", "512"))
            dense = int(self.cfg.idx_candidate_topk)
            for r0 in range(max(A, self.pc_min), dense, A):
                for b in m.PF_BUCKETS:
                    T = min(b - 1, dense - r0)
                    if next(x for x in m.PF_BUCKETS if x >= T) != b or r0 + b + 64 > m.max_ctx:
                        continue
                    base = rnd(r0 + 1)
                    self.prefill(0, base)
                    self.prefill(0, base[:r0] + rnd(T))
                    m.warm_head_blocks(b)
            # past the dense regime: a full chunk at group offset 768 (slabs of 256 groups; the dense ones above compile
            # the 128-group slabs, smaller chunks write one slab of their size)
            r0, T = dense + 1024, self.chunk - 1
            if int(os.environ.get("HY4_PC_ALIGN", "512")) <= 1024 and r0 + T + 64 < m.max_ctx:
                base = rnd(r0 + 1)
                self.prefill(0, base)
                self.prefill(0, base[:r0] + rnd(T))
        # packed prefills: every (B, Tb) plan up to the slot count (their batched attention / segment slices)
        for B in (2, 4, 8, 16):
            for Tb in m.PF_BUCKETS:
                if B <= self.S and m.pack_plan([Tb - 1] * B) == (B, Tb):
                    self.prefill_many([(s, rnd(Tb - 1), None) for s in range(B)])
                    m.warm_head_blocks(B * Tb)
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
        # the single path's larger score buckets (slot 0 selects): a context past the first bucket compiled at serve time
        for b in dec.buckets[1:] if solo else []:
            dec.min_bucket = b
            self.step(solo[0])
        dec.min_bucket = 0
        self._warm_spec(solo)
        dec.min_width = 0
        # dense-only passes (no row selects: no scoring) at every batch size: slot 0 short again
        self.prefill(0, rnd(int(prompt_len or 64)))
        for R in range(2, self.S + 1):
            self.step({s: (None, None, False) for s in range(R)})
        m.prefill_caches_mark()  # (the warm-up prefills above ran before any capture: their cached constants stay)
        dec.trace = self.trace
        logger.info(f"warm: eager batched passes done; {m.dram_report()}")
        for _ in range(2):  # spec first: the drafter's row slices compile at its first traced draft (a trace release)
            self._warm_spec(solo)
        steps = int(os.environ.get("HY4_WARM_STEPS", "3"))
        for R in range(1, self.S + 1):
            for _ in range(steps):
                self.step({s: (None, None, False) for s in range(R)})
        for r in solo[:2]:  # capture the single-path traces of both regimes
            for _ in range(steps):
                self.step(r)
        for b in dec.buckets[1:] if solo else []:  # and of every larger score bucket
            dec.min_bucket = b
            for _ in range(steps):
                self.step(solo[0])
        dec.min_bucket = 0
        for s in range(self.S):
            self.finish(s)
            m.set_slot(s)
            m.reset()
        m.set_slot(0)
        self.kept.clear()
        logger.info(f"warm: traces captured; {m.dram_report()}")
        self.stats = {k: 0 if isinstance(v, int) else 0.0 for k, v in self.stats.items()}
        logger.info(f"Hy4 batch engine warm-up {time.time() - t0:.1f} s ({self.S} slots, widths {widths}, "
                    f"{len(dec.btraces)} batched traces, {len(dec.traces)} single)")

    def _warm_spec(self, solo):
        """k > 0: the solo path's verify passes (k + 1 rows) and drafts, both regimes (slot 0 selects, slot 1 dense)
        and every larger score bucket (slot 0): per bucket a 1-row pass (re-entering solo), then two verify passes."""
        if not (self.k and solo):
            return
        dec = self.dec
        for b in [0] + list(dec.buckets[1:]):
            dec.min_bucket = b
            for r in (solo[:2] if b == 0 else solo[:1]):
                if self.solo in r:  # leave and re-enter solo: the first pass is a 1-row pass
                    self.step(solo[1] if len(solo) > 1 and r is solo[0] else solo[0])
                for _ in range(3):
                    self.step(r)
        dec.min_bucket = 0

    def capture(self):
        """Batched traces are captured on first use per (rows, width) by tt/decode_batch.Hy4BatchDecoder."""
