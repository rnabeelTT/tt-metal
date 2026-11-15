# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-user generation engine for Inkling: chunked prefill, then speculative decoding with the native MTP drafter
(k drafts per verify pass; k = 0 is plain decode). The vLLM adapter (tt/generator_vllm.py) drives it as its
speculative step object; tools can drive it directly (generate()).

One step (k drafts d_1..d_k pending from the previous step, last emitted token x at position p):
  verify   the target on rows [x, d_1 .. d_k] at positions p .. p + k (rows mode, conv states held), one logits row
           each; greedy: n = the leading drafts equal to the argmax rows, next = argmax row n; sampled: exact
           speculative sampling (min(1, p / q) accept, residual resample: M2.7's SpecSampler)
  commit   the target's conv states advance past the n + 1 accepted rows
  draft    MTP depth 0 over the n + 1 accepted rows (the target's normed hidden of row j, the token after it), its last
           row -> d'_1; depths 1 .. k - 1 chain one row each -> d'_2 .. d'_k
Emits d_1 .. d_n, next. The output equals plain greedy decode token for token (tools/spec_bench.py checks it).

Traces (trace=True): every verify width 1 .. k + 1 and its commits (n = 1 .. R), the drafter's depth 0 per (verify
width, accepted rows) and each deeper depth. The hybrid MoE's host step runs between trace segments (hybrid/seg.py).
Rules learned in the dev harness (each was a real bug): run one eager pass at every row width AND the drafter at
every width / depth BEFORE any capture, then capture everything up front -- a device tensor a layer allocates lazily
after a capture can sit in that trace's freed intermediates and is overwritten by its replays; the drafter runs on
its own CCL semaphores (mtp.own_ccl); hybrid needs HYB_DECODE_ROWS >= the verify width; logits are untilized on the
device before the readback (a tile readback moves 32 padded rows).

KV: the 7 global layers keep their own paged caches (identity page table), the sliding layers their rings, the MTP
depths theirs; vLLM's KV allocation is unused (single user, no prefix caching). Every prefill starts a new sequence.
"""

from __future__ import annotations

import os
import time

import torch
from loguru import logger

import ttnn
from models.autoports.minimaxai_minimax_m2_7.tt.mtp_sampling import SpecSampler
from models.autoports.thinkingmachines_inkling_small.tt.inkling_mtp import InklingMtp

ROWS_WARM = 8  # widest row group warmed eagerly (hybrid per-width buffers exist for 1..8 before any capture)


def prefill_buckets(chunk: int):
    """Row counts a (last) prefill chunk is padded to: 128, 256, ... up to the chunk."""
    out, b = [], 128
    while b < chunk:
        out.append(b)
        b *= 2
    return out + [chunk]


class InklingSpecServing:
    """The engine; also the vLLM adapter's speculative step object (`step` / `warm` / `capture` / `k`)."""

    multi = False  # the plugin-facing flags the shared adapter code reads
    adaptive = False
    in_trace = False

    def __init__(self, model, args, mesh, k: int, trace: bool = True):
        self.model, self.args, self.mesh = model, args, mesh
        self.k = int(k)
        self.trace = bool(trace)
        self.rep = ttnn.ReplicateTensorToMesh(mesh)
        self.V = int(args.unpadded_vocab_size)
        self.mup = float(args.logits_mup_width_multiplier)
        att = model.layers[0].attention
        self.max_seq_len = int(args.max_seq_len)
        self.chunk = int(att.prefill_chunk)
        self.buckets = prefill_buckets(self.chunk)
        self.pt_host = torch.arange(att.max_blocks, dtype=torch.int32)
        self.pt = ttnn.from_torch(self.pt_host.reshape(1, -1), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                  device=mesh, mesh_mapper=self.rep)
        self.mtp = InklingMtp(model, self.k) if self.k else None
        self.sconvs = [c for layer in model.layers
                       for c in (layer.attention.k_sconv, layer.attention.v_sconv, layer.attn_sconv, layer.mlp_sconv)]
        eos = args.inkling.get("eos_token_id", 200006)
        self.eos = set(eos) if isinstance(eos, (list, tuple)) else {int(eos)}
        # verify traces: R -> persistent inputs (ids, positions, kv index), trace id, outputs; commits (R, n) -> id
        self.bufs, self.tid, self.out, self.keep, self.ctid = {}, {}, {}, {}, {}
        self.last_R = None  # the width of the last traced verify (its commit traces read that capture's conv history)
        self.last_traced = False
        # drafter traces
        self.t0, self.tj = {}, {}
        self.d_ids = self.d_pos = self.prev = None
        self.captured = False
        # the continuation: the next token to verify, its position, the pending drafts (+ their q when sampling)
        self._tok = self._pos = None
        self._drafts, self._qs = [], None
        self._smp = SpecSampler(0.0)
        self.stats = {"steps": 0, "emitted": 0, "t_verify": 0.0, "t_draft": 0.0, "t_prefill": 0.0, "prefills": 0}

    # ------------------------------------------------------------------ device inputs
    def _ids_host(self, ids):
        return ttnn.from_torch(torch.tensor(ids, dtype=torch.int32).reshape(1, len(ids)), dtype=ttnn.uint32,
                               layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.rep)

    def _pos_host(self, p0, n):
        return ttnn.from_torch(torch.arange(p0, p0 + n, dtype=torch.int32), dtype=ttnn.int32,
                               layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.rep)

    @staticmethod
    def _idx_torch(p0, n):
        """kv_rows_update's [n, n] index: row u -> position p0 + u on the diagonal, -1 elsewhere."""
        m = torch.full((n, n), -1, dtype=torch.int32)
        m[torch.arange(n), torch.arange(n)] = torch.arange(p0, p0 + n, dtype=torch.int32)
        return m

    def _idx_host(self, p0, n):
        return ttnn.from_torch(self._idx_torch(p0, n), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                               mesh_mapper=self.rep)

    def _dev(self, host_t):
        return ttnn.to_device(host_t, self.mesh)

    @staticmethod
    def _rm_logits(lg):
        """Logits untilized on the device (same values): a tile readback moves all 32 padded rows."""
        if lg.is_sharded():
            lg = ttnn.sharded_to_interleaved(lg, ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.untilize_with_unpadding(lg, [int(d) - 1 for d in lg.shape])

    def _read(self, lg, R):
        return ttnn.to_torch(ttnn.get_device_tensors(lg)[0]).float().reshape(-1, lg.shape[-1])[:R, : self.V]

    # ------------------------------------------------------------------ target: verify rows + commits
    def _body(self, di, dp, dk, R):
        model = self.model
        x = model.embd(di)
        x = ttnn.reshape(x, (1, 1, R, x.shape[-1]))
        for layer in model.layers:
            x = layer.forward_rows(x, dp, self.pt, commit_all=False, kv_idx=dk)
        h = model._final_norm_decode(x)  # replicated, normed (the model's 1/16 folded into the norm weight)
        # the drafter wants final_norm(h) without that 1/16: x16
        return self._rm_logits(model._lm_head(h)), ttnn.multiply(h, self.mup)

    def _rows_eager(self, ids, p0):
        """Eager rows pass: -> (logits [R, V] fp32 host, normed hidden x mup [1, 1, R, H] device)."""
        R = len(ids)
        lg, h = self._body(self._dev(self._ids_host(ids)), self._dev(self._pos_host(p0, R)),
                           self._dev(self._idx_host(p0, R)), R)
        self.last_traced = False
        return self._read(lg, R), h

    def _commit_eager(self, n):
        for layer in self.model.layers:
            layer.commit_rows(n)

    def _capture_verify(self, R):
        di, dp, dk = self.bufs[R]
        tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        out = self._body(di, dp, dk, R)
        ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
        self.tid[R], self.out[R] = tid, out
        self.keep[R] = [c._last_seq for c in self.sconvs]  # trace-owned conv history the commit traces read
        for n in range(1, R + 1):
            t = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            self._commit_eager(n)
            ttnn.end_trace_capture(self.mesh, t, cq_id=0)
            self.ctid[(R, n)] = t

    def _verify(self, ids, p0):
        """Verify rows: traced when this width was captured, else eager. -> (logits [R, V] host, hidden x mup)."""
        R = len(ids)
        if R not in self.tid:
            return self._rows_eager(ids, p0)
        di, dp, dk = self.bufs[R]
        ttnn.copy_host_to_device_tensor(self._idx_host(p0, R), dk)
        ttnn.copy_host_to_device_tensor(self._ids_host(ids), di)
        ttnn.copy_host_to_device_tensor(self._pos_host(p0, R), dp)
        ttnn.execute_trace(self.mesh, self.tid[R], cq_id=0, blocking=False)
        self.last_R, self.last_traced = R, True
        lg, hm = self.out[R]
        return self._read(lg, R), hm

    def _commit(self, n):
        if self.last_traced:
            ttnn.execute_trace(self.mesh, self.ctid[(self.last_R, n)], cq_id=0, blocking=False)
        else:
            self._commit_eager(n)

    # ------------------------------------------------------------------ drafter
    def _pick(self, lg, smp, qs):
        row = self._read(lg, 1)[0]
        if smp.greedy:
            return int(row.argmax())
        q = smp.probs(row)
        qs.append(q)
        return smp.sample(q)

    def _draft_eager(self, hrows, toks, p0, k, smp):
        mtp, model = self.mtp, self.model
        R = len(toks)
        dp = mtp.depths[0]
        e = model.embd(self._dev(self._ids_host(toks)))
        e = mtp._gather(ttnn.reshape(e, (1, 1, R, e.shape[-1])))
        x_in = ttnn.concat([mtp._norm(hrows, dp.hn), mtp._norm(e, dp.en)], dim=-1)
        out = dp.block.forward_rows(ttnn.linear(x_in, dp.w_in), self._dev(self._pos_host(p0, R)), self.pt,
                                    commit_all=True)
        full = mtp._gather(out)
        prev = ttnn.slice(full, (0, 0, R - 1, 0), (1, 1, R, full.shape[-1]))
        drafts, qs = [], []
        d = self._pick(model._lm_head(ttnn.multiply(prev, mtp.inv_mup)), smp, qs)
        drafts.append(d)
        for j in range(1, k):
            lg, prev = mtp.step(j, prev, self._dev(self._ids_host([d])), self._dev(self._pos_host(p0 + R - 1 + j, 1)),
                                self.pt)
            d = self._pick(lg, smp, qs)
            drafts.append(d)
        return drafts, (None if smp.greedy else qs)

    def _d_depth0(self, R_src, w):
        mtp, model = self.mtp, self.model
        dp = mtp.depths[0]
        h = self.out[R_src][1]
        hrows = ttnn.slice(h, (0, 0, 0, 0), (1, 1, w, h.shape[-1]))
        e = model.embd(self.d_ids[w])
        e = mtp._gather(ttnn.reshape(e, (1, 1, w, e.shape[-1])))
        x_in = ttnn.concat([mtp._norm(hrows, dp.hn), mtp._norm(e, dp.en)], dim=-1)
        out = dp.block.forward_rows(ttnn.linear(x_in, dp.w_in), self.d_pos[w], self.pt, commit_all=True)
        full = mtp._gather(out)
        last = ttnn.slice(full, (0, 0, w - 1, 0), (1, 1, w, full.shape[-1]))
        lg = self._rm_logits(model._lm_head(ttnn.multiply(last, mtp.inv_mup)))
        ttnn.copy(ttnn.typecast(last, ttnn.bfloat16) if last.dtype != ttnn.bfloat16 else last, self.prev)
        return lg

    def _d_depth(self, j):
        lg, full = self.mtp.step(j, self.prev, self.d_ids[1], self.d_pos[1], self.pt)
        ttnn.copy(ttnn.typecast(full, ttnn.bfloat16) if full.dtype != ttnn.bfloat16 else full, self.prev)
        return self._rm_logits(lg)

    def _d_capture(self, f):
        f()  # eager first: compiles every program (a capture cannot)
        tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        out = f()
        ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
        return tid, out

    def _draft(self, R_src, toks, p0, k, smp):
        """k drafts after the accepted rows toks (the tokens following verify rows 0 .. w - 1 of the last pass)."""
        w = len(toks)
        if not self.t0 or not self.last_traced or (R_src, w) not in self.t0:
            hrows = ttnn.slice(self._last_h, (0, 0, 0, 0), (1, 1, w, self._last_h.shape[-1]))
            return self._draft_eager(hrows, toks, p0, k, smp)
        ttnn.copy_host_to_device_tensor(self._ids_host(toks), self.d_ids[w])
        ttnn.copy_host_to_device_tensor(self._pos_host(p0, w), self.d_pos[w])
        tid, lg = self.t0[(R_src, w)]
        ttnn.execute_trace(self.mesh, tid, cq_id=0, blocking=False)
        drafts, qs = [], []
        d = self._pick(lg, smp, qs)
        drafts.append(d)
        for j in range(1, k):
            ttnn.copy_host_to_device_tensor(self._ids_host([d]), self.d_ids[1])
            ttnn.copy_host_to_device_tensor(self._pos_host(p0 + w - 1 + j, 1), self.d_pos[1])
            tid, lg = self.tj[j]
            ttnn.execute_trace(self.mesh, tid, cq_id=0, blocking=False)
            d = self._pick(lg, smp, qs)
            drafts.append(d)
        return drafts, (None if smp.greedy else qs)

    # ------------------------------------------------------------------ warm-up / capture
    def _reset(self):
        self.model.reset_sequence_state()
        if self.mtp is not None:
            self.mtp.reset()

    def warm(self, prompt_len=None):
        """Every program a request can run, eagerly, before any capture: each prefill bucket (a warm-up prompt of
        `prompt_len` tokens covers the full chunks too), the verify rows at every width 1..8, the drafter at every
        accepted-row width and depth. Then the sequence state is reset."""
        t0 = time.time()
        ids = torch.arange(1000, 1000 + max(self.buckets) + 64, dtype=torch.int64) % 200000
        for b in self.buckets:  # one prefill per bucket size (a b-row prompt pads to bucket b)
            self._reset()
            self.model.prefill_tokens(ids[:b], self.pt_host, buckets=self.buckets)
        p = self.buckets[-1]
        smp = SpecSampler(0.0)
        for R in range(1, ROWS_WARM + 1):
            _, h = self._rows_eager([7] * R, p)
            if self.mtp is not None and R <= self.k + 1:
                with self.mtp.own_ccl():
                    self._draft_eager(ttnn.slice(h, (0, 0, 0, 0), (1, 1, R, h.shape[-1])), [7] * R, p, self.k, smp)
            for n in range(1, R + 1):  # the commit programs (state restored by the reset below)
                self._commit_eager(n)
        self._reset()
        logger.info(f"Inkling engine warm-up: {time.time() - t0:.1f}s (prefill buckets {self.buckets}, "
                    f"rows 1..{ROWS_WARM}, drafter k={self.k})")

    def capture(self):
        """Every verify width 1 .. k + 1 with its commits, then the drafter (depth 0 per (width, accepted rows), each
        deeper depth). The eager passes here advance the states: reset after."""
        if not self.trace or self.captured:
            return
        t0 = time.time()
        p = self.buckets[-1]
        for R in range(1, self.k + 2):
            self.bufs[R] = (self._dev(self._ids_host([0] * R)), self._dev(self._pos_host(0, R)),
                            self._dev(self._idx_host(0, R)))
        if self.mtp is not None:
            H = int(self.args.dim)
            self.prev = ttnn.from_torch(torch.zeros(1, 1, 1, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                        device=self.mesh, mesh_mapper=self.rep)
            self.d_ids = {w: self._dev(self._ids_host([0] * w)) for w in range(1, self.k + 2)}
            self.d_pos = {w: self._dev(self._pos_host(0, w)) for w in range(1, self.k + 2)}
        for R in range(1, self.k + 2):
            di, dp, dk = self.bufs[R]
            ttnn.copy_host_to_device_tensor(self._idx_host(p, R), dk)
            ttnn.copy_host_to_device_tensor(self._pos_host(p, R), dp)
            self._body(di, dp, dk, R)  # eager once more right before the capture (programs cached, buffers warm)
            self._capture_verify(R)
        if self.mtp is not None:
            with self.mtp.own_ccl():
                self.t0[(1, 1)] = self._d_capture(lambda: self._d_depth0(1, 1))
                for kk in range(1, self.k + 1):
                    for w in range(1, kk + 2):
                        self.t0[(kk + 1, w)] = self._d_capture(lambda kk=kk, w=w: self._d_depth0(kk + 1, w))
                for j in range(1, self.k):
                    self.tj[j] = self._d_capture(lambda j=j: self._d_depth(j))
        self.captured = True
        self._reset()
        logger.info(f"Inkling engine captures: {time.time() - t0:.1f}s (verify widths 1..{self.k + 1}, "
                    f"{len(self.ctid)} commits, {len(self.t0) + len(self.tj)} drafter traces)")

    # ------------------------------------------------------------------ serving
    def allocate_kv(self, kv_shape, dtype):
        """The adapter hook: the engine's caches are its own (see the module doc)."""

    def warm_prefill_path(self):
        """The adapter hook: warm() covers the prefill programs."""

    def _first(self, last_tok, p, smp):
        """The first verify pass of a sequence (the last prompt token, no drafts) and the drafts after it."""
        L, h = self._verify([last_tok], p)
        self._last_h = h
        self.last_logits = L[0]
        nxt = int(L[0].argmax()) if smp.greedy else smp.sample(smp.probs(L[0]))
        self._commit(1)
        self._drafts, self._qs = [], None
        if self.mtp is not None and nxt not in self.eos:
            with self.mtp.own_ccl():
                self._drafts, self._qs = self._draft(1, [nxt], p, self.k, smp)
        self._tok, self._pos = nxt, p + 1
        return nxt

    def prefill(self, prompt_ids, sampling=None) -> int:
        """A new sequence: prefill the prompt (all but its last token; that one is the first verify row) and return
        the first generated token. sampling = (temperature, top_k, top_p, seed) or None (greedy)."""
        t0 = time.time()
        P = len(prompt_ids)
        if P < 1:
            raise ValueError("empty prompt")
        if P + self.k + 1 > self.max_seq_len:
            raise ValueError(f"prompt of {P} tokens + the verify window exceeds the context ({self.max_seq_len})")
        self._smp = self._sampler(sampling, P)
        self._reset()
        if P > 1:
            self.model.prefill_tokens(torch.as_tensor(prompt_ids[:-1]), self.pt_host, buckets=self.buckets)
        nxt = self._first(int(prompt_ids[-1]), P - 1, self._smp)
        self.stats["t_prefill"] += time.time() - t0
        self.stats["prefills"] += 1
        return nxt

    @staticmethod
    def _sampler(sampling, salt):
        if sampling is None:
            return SpecSampler(0.0)
        t, k, p, seed = sampling
        return SpecSampler(float(t), int(k or 0), float(p if p is not None else 1.0),
                           seed=(int(seed) if seed is not None else None))

    def step(self, tok: int = None, pos: int = None, page_table_row=None, host_authoritative=False,
             sampling=None) -> list[int]:
        """One verify pass: -> the accepted drafts followed by the verified next token (1 .. k + 1 ids). The engine
        owns the continuation (on the plugin's async path its host token can lag a step); the plugin's tok / pos are
        checked when it flags them authoritative."""
        if self._pos is None:
            raise RuntimeError("Inkling step before any prefill")
        if host_authoritative and tok is not None and (int(tok), int(pos)) != (self._tok, self._pos):
            if int(pos) != self._pos:
                raise RuntimeError(f"Inkling step input (tok {tok}, pos {pos}) differs from the engine's continuation "
                                   f"(tok {self._tok}, pos {self._pos})")
            # the plugin sampled the token itself (host sampling after a logits prefill): adopt it, drop the drafts
            self._tok, self._drafts, self._qs = int(tok), [], None
            self.stats["reanchors"] = self.stats.get("reanchors", 0) + 1
        x, p, drafts, qs, smp = self._tok, self._pos, self._drafts, self._qs, self._smp
        if p + len(drafts) + 1 > self.max_seq_len:  # the context's last positions: no drafts past it
            drafts = drafts[: max(0, self.max_seq_len - p - 1)]
            qs = qs[: len(drafts)] if qs is not None else None
        rows = [x] + drafts
        a = time.time()
        L, h = self._verify(rows, p)
        self._last_h = h
        if smp.greedy:
            n = 0
            am = L.argmax(-1).tolist()
            while n < len(drafts) and drafts[n] == am[n]:
                n += 1
            nxt = am[n]
        else:
            n, nxt = smp.accept(L, drafts, qs)
        self._commit(n + 1)
        b = time.time()
        new = drafts[:n] + [nxt]
        hit = next((i for i, t in enumerate(new) if t in self.eos), None)
        if hit is not None:
            new = new[: hit + 1]
        self._drafts, self._qs = [], None
        if self.mtp is not None and hit is None and p + n + 1 < self.max_seq_len - 1:
            with self.mtp.own_ccl():
                self._drafts, self._qs = self._draft(len(rows), rows[1 : n + 1] + [nxt], p, self.k, smp)
        self._tok, self._pos = nxt, p + n + 1
        st = self.stats
        st["steps"] += 1
        st["emitted"] += len(new)
        st["t_verify"] += b - a
        st["t_draft"] += time.time() - b
        if st["steps"] % 512 == 0:
            logger.info(f"Inkling spec: {st['emitted'] / st['steps']:.2f} tokens/step over {st['steps']} steps, "
                        f"verify {1e3 * st['t_verify'] / st['steps']:.1f} ms, draft "
                        f"{1e3 * st['t_draft'] / st['steps']:.1f} ms per step")
        return new

    def generate(self, prompt_ids, max_new: int, sampling=None):
        """Standalone use: -> generated ids (stops at the end of turn)."""
        out = [self.prefill(prompt_ids, sampling)]
        while len(out) < max_new and out[-1] not in self.eos:
            out += self.step()
        return out[:max_new]
