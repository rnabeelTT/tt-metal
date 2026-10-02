# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Traced MTP step for ONE user (v2): four fixed-shape device programs replayed per pass, tiny host writes / reads between.

    pass   : snapshot the GDN state -> the model over BW = 8 rows [t, d_1..d_k, pad] (rows mode) -> final norm -> LM head
             -> per-shard argmax / max of every row (read back: 2 x [BW] per device); the pass's final-normed hidden
             rows and every GDN layer's 128-row normed input are copied into persistent buffers.
    commit : restore the snapshot, replay the 36 GDN layers on the saved inputs with valid_len = a + 1 (device masks,
             `PrefillLen`); skipped when every draft was accepted (the pass state is then exact).
    draft  : one MTP-head row (token, hidden, position) -> per-shard argmax / max (read back) and the head's normed
             hidden copied into the next step's hidden input.
    rows   : the head over BW rows (tokens rows[1..], hidden = the pass's rows) writing its K/V (rows past the accepted
             count land on positions that later real rows overwrite before anything reads them).

Every program is compiled by an eager warm run BEFORE any capture (#48536: a program first compiled after a capture can
have its binaries overwritten by the next replay); the only per-pass host work is a few small `copy_host_to_device_tensor`
writes, the `PrefillLen.stage` mask writes and the argmax read-backs. Host greedy accept (v3: device accept).
"""
from __future__ import annotations

import os
import time

import torch

import ttnn
from models.autoports.qwen_qwen3_coder_next.tt.mtp_rows import (
    GDN_CHUNK,
    layer_forward_rows,
    rebuild_conv_carry,
    rows_pos_masks_host,
    rows_width,
)
from models.autoports.qwen_qwen3_coder_next.tt.prefill_len import PrefillLen

_DRAM = ttnn.DRAM_MEMORY_CONFIG


def _host_u32(vals, shape):
    t = torch.zeros(*shape, dtype=torch.int32)
    flat = t.reshape(-1)
    flat[: len(vals)] = torch.tensor(vals, dtype=torch.int32)
    return ttnn.from_torch(t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)


def _host_i32(vals, shape):
    t = torch.zeros(*shape, dtype=torch.int32)
    flat = t.reshape(-1)
    flat[: len(vals)] = torch.tensor(vals, dtype=torch.int32)
    return ttnn.from_torch(t, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)


class MtpTraced:
    def __init__(self, gen, head, k: int, bw: int | None = None):
        bw = bw or rows_width(k + 1)
        assert 1 <= k < bw <= 32
        self.gen, self.head, self.k, self.BW = gen, head, int(k), int(bw)
        self.model = gen.model
        m = self.model
        self.mesh = m.mesh_device
        self.V = int(gen.args.vocab_size)
        self.H = int(gen.args.dim)
        self.page_table = gen.page_table[:1]  # torch [1, blocks]
        self.dn_layers = [layer.attention for layer in m.layers if not layer.is_full_attention]
        self.gdn_layers = [layer for layer in m.layers if not layer.is_full_attention]
        K = self.dn_layers[0].K
        dev = lambda t: ttnn.to_device(t, self.mesh, memory_config=_DRAM)
        BW = self.BW
        # ---- pass inputs (persistent): tokens [1,1,1,S], positions [BW], rot idxs [1,BW], page table [BW, blocks]
        tok, pos, rot, pt = m.prepare_decode_inputs_host(
            torch.zeros(BW, dtype=torch.int32), torch.zeros(BW, dtype=torch.int32), self.page_table.repeat(BW, 1)
        )
        self.p_tok, self.p_pos, self.p_rot, self.p_pt = dev(tok), dev(pos), dev(rot), dev(pt)
        self.p_masks = dev(rows_pos_masks_host([0] * BW, BW))  # per-row K/V write positions (-1 = skip)
        self.plen = PrefillLen(self.mesh, GDN_CHUNK, K)  # valid rows of the pass (R)
        self.plen_commit = PrefillLen(self.mesh, GDN_CHUNK, K)  # accepted rows (a + 1)
        # ---- persistent pass outputs / commit inputs
        self.saved = [{} for _ in self.gdn_layers]  # per GDN layer: persistent scan inputs (filled at the warm run)
        self.snap_rec = [ttnn.clone(dn.rec_state) for dn in self.dn_layers]
        self.snap_conv = [[ttnn.clone(c) for c in dn.conv_states] for dn in self.dn_layers]
        self.h_out = None  # [1,1,BW,H] replicated
        # ---- head inputs
        tok1, pos1, rot1, pt1 = m.prepare_decode_inputs_host(
            torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32), self.page_table
        )
        self.d_tok, self.d_pos, self.d_rot, self.d_pt = dev(tok1), dev(pos1), dev(rot1), dev(pt1)
        self.d_hid = ttnn.from_torch(
            torch.zeros(1, 1, 1, self.H, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=_DRAM,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        tokr, posr, rotr, ptr = m.prepare_decode_inputs_host(
            torch.zeros(BW, dtype=torch.int32), torch.zeros(BW, dtype=torch.int32), self.page_table.repeat(BW, 1)
        )
        self.r_tok, self.r_pos, self.r_rot, self.r_pt = dev(tokr), dev(posr), dev(rotr), dev(ptr)
        # one-hot row selector for "the hidden of accepted row a becomes the next draft's hidden"
        self.sel = ttnn.from_torch(
            torch.zeros(1, 1, 1, BW, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=_DRAM,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        self.traces = {}
        self.outs = {}
        self.timing = {
            "pass": [],
            "commit": [],
            "draft": [],
            "rows": [],
            "ids_read": [],
            "lg_read": [],
            "cands_read": [],
        }
        self._last_logits = None  # [B, V] host logits of the last pass / draft read back in sampling mode
        self._last_cands = None  # (values, ids) [B, n] of the last candidate read (sparse sampling)
        self.rope_delta = 0  # the request's M-RoPE position delta (image prompts): rope position = KV position + delta
        self._cand_half = None

    # ------------------------------------------------------------------------------------------------ programs
    def _argmax_max(self, logits):
        """Per-shard argmax + max over the vocab dim of [1,1,B,V/TP] (read back by the host, combined across shards).
        Also returns the ROW_MAJOR logits (the sampling mode reads them back: unpadded rows, 0.5 MB per row per mesh,
        where the TILE tensor would transfer its 32-row padding)."""
        # argmax on a TILE input is single-core (9.2 ms for 8 x 62080!); the ROW_MAJOR path is multi-core
        lg_rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        am = ttnn.argmax(lg_rm, dim=-1, keepdim=True)  # [1,1,B,1] uint32
        if os.environ.get("QWEN35_MTP_ARGMAX_MODE", "gather") == "gather":
            # the winning value by a gather at the argmax index (ttnn.max over the tile logits cost 244 us)
            mx = ttnn.gather(lg_rm, -1, am)  # [1,1,B,1] (gather takes UINT32 / UINT16 indices; argmax gives UINT32)
        else:
            mx = ttnn.max(logits, dim=-1, keepdim=True)  # [1,1,B,1]
        return am, mx, lg_rm

    def _pass_body(self):
        m = self.model
        BW = self.BW
        for dn, rec, taps in zip(self.dn_layers, self.snap_rec, self.snap_conv):
            ttnn.copy(dn.rec_state, rec)
            for c, t in zip(dn.conv_states, taps):
                ttnn.copy(c, t)
        cos, sin = m._rope_from_idxs(self.p_rot)
        tok = m._decode_tokens_view(self.p_tok, BW)
        x = m.embd(tok)
        x = ttnn.reshape(x, (1, 1, BW, x.shape[-1]))
        x = ttnn.to_memory_config(x, _DRAM)
        gi = 0
        for layer in m.layers:
            # GDN layers copy their scan inputs into self.saved[gi] (persistent after the first warm run)
            y = layer_forward_rows(
                layer,
                x,
                self.plen,
                cos=cos,
                sin=sin,
                position_tensor=self.p_pos,
                page_table=self.p_pt,
                saved=self.saved,
                rows_pos_masks=self.p_masks,
                gdn_index=gi,
            )
            if not layer.is_full_attention:
                gi += 1
            ttnn.deallocate(x)
            x = y
        h = m._final_norm_decode(x)
        ttnn.deallocate(x)
        logits = ttnn.linear(h, m.lm_head_weight, compute_kernel_config=m._lm_head_ckc())
        if self.h_out is None:
            self.h_out = h
        else:
            ttnn.copy(h, self.h_out)
            ttnn.deallocate(h)
        am, mx, lg_rm = self._argmax_max(logits)
        for t in (cos, sin):
            ttnn.deallocate(t)
        # outputs: the ROW_MAJOR logits (the speculative sampler's full-row fallback reads them back) and the TILE
        # logits (input of the candidate trace: per-shard top-k for the sampled path with a top_k <= 32)
        return am, mx, lg_rm, logits

    def _commit_body(self):
        for dn, rec, taps in zip(self.dn_layers, self.snap_rec, self.snap_conv):
            ttnn.copy(rec, dn.rec_state)
            for t, c in zip(taps, dn.conv_states):
                ttnn.copy(t, c)
        for layer, sink in zip(self.gdn_layers, self.saved):
            dn = layer.attention
            rebuild_conv_carry(dn)
            dn.commit_scan(sink, self.plen_commit)  # conv FIR (taps) + chunk scan only
        return ()

    def _draft_body(self):
        lg, h_next = self.head.decode_step(self.d_tok, self.d_hid, self.d_pos, self.d_rot, self.d_pt)
        am, mx, lg_rm = self._argmax_max(lg)
        ttnn.copy(h_next, self.d_hid)
        ttnn.deallocate(h_next)
        return am, mx, lg_rm, lg

    # per-shard candidate count of the device top-k (2 halves x 32): SpecSampler.SPARSE_TOP_K must not exceed 32
    CAND_K = 32

    def _cands_body(self, logits):
        """Per-shard top-k candidates of TILE logits [1,1,B,W]: W padded to the next power of two and split in two
        halves of <= 32768 (the multi-core top-k bound), top-32 of each -> (values, ids) x 2, ids local to the half.
        Shared with models/common/modules/sampling/sampling_1d.py (same pad / split shape)."""
        import sys

        W = logits.shape[-1]
        Wp = 1 << (W - 1).bit_length()
        x = logits
        if Wp != W:
            x = ttnn.pad(logits, [(0, 0), (0, 0), (0, 0), (0, Wp - W)], value=-sys.float_info.max)
        halves = ttnn.split(x, Wp // 2, dim=3) if Wp > 32768 else [x]
        if x is not logits:
            ttnn.deallocate(x)
        outs = []
        for h in halves:
            v, i = ttnn.topk(h, k=self.CAND_K, dim=-1)
            outs += [v, i]
            if h is not logits:
                ttnn.deallocate(h)
        self._cand_half = Wp // 2 if Wp > 32768 else Wp
        return tuple(outs)

    def _rows_body(self):
        att = self.head.layer.attention
        att.rows_kv_write = True
        try:
            lg, h = self.head.decode_step(self.r_tok, self.h_out, self.r_pos, self.r_rot, self.r_pt)
        finally:
            att.rows_kv_write = False
        ttnn.deallocate(lg)
        ttnn.deallocate(h)
        return ()

    def _select_hidden(self, a: int):
        """d_hid <- h_out[row a] via a one-hot matmul (fixed program; a variable slice would compile per value)."""
        sel = torch.zeros(1, 1, 1, self.BW, dtype=torch.bfloat16)
        sel[0, 0, 0, a] = 1.0
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(sel, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), self.sel)
        picked = ttnn.matmul(self.sel, self.h_out, memory_config=_DRAM)  # [1,1,1,H]
        ttnn.copy(picked, self.d_hid)
        ttnn.deallocate(picked)

    # ------------------------------------------------------------------------------------------------ warm / capture
    def warm(self, flush=None):
        """Eager run of every program once (compiles them; allocates the persistent outputs). `flush()` (optional) runs
        between programs (the Tracy tool reads the device profiler there: its buffers hold a few hundred ops)."""
        self.plen.stage(self.k + 1)
        self.plen_commit.stage(1)
        # the pass runs twice: the first run adopts its outputs as the persistent buffers, the second compiles the copies
        # into them (a program first compiled inside the capture would TT_FATAL: "Cannot load new binaries during trace
        # capture")
        bodies = (
            ("pass", self._pass_body),
            ("pass", self._pass_body),
            ("commit", self._commit_body),
            ("draft", self._draft_body),
            ("rows", self._rows_body),
        )
        for name, body in bodies:
            outs = body()
            if name in ("pass", "draft"):
                for t in self._cands_body(outs[3]):
                    ttnn.deallocate(t)
            for t in outs:
                ttnn.deallocate(t)
            if flush:
                flush()
        self._select_hidden(0)
        ttnn.synchronize_device(self.mesh)

    def capture(self, flush=None):
        for name, body in (
            ("pass", self._pass_body),
            ("commit", self._commit_body),
            ("draft", self._draft_body),
            ("rows", self._rows_body),
        ):
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            outs = body()
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            self.traces[name] = tid
            self.outs[name] = outs
            if flush:
                flush()
        for name in ("pass", "draft"):
            # the sampled path's candidates: a separate trace (greedy passes never pay for it) reading the TILE logits
            # the main trace left in its persistent output
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            outs = self._cands_body(self.outs[name][3])
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            self.traces["cands_" + name] = tid
            self.outs["cands_" + name] = outs
            if flush:
                flush()
        ttnn.synchronize_device(self.mesh)

    def release(self):
        for tid in self.traces.values():
            ttnn.release_trace(self.mesh, tid)
        self.traces = {}

    # ------------------------------------------------------------------------------------------------ per-pass host side
    _BODIES = {"pass": "_pass_body", "commit": "_commit_body", "draft": "_draft_body", "rows": "_rows_body"}

    def _run(self, name, want_logits=False, want=None):
        """Replay program `name`; returns the per-row argmax ids. `want` = "rows": also read the ROW_MAJOR logits rows
        back into _last_logits; "cands": run the candidate trace and read the per-shard top-k into _last_cands (falls
        back to "rows" while the programs run eagerly). `want_logits=True` is the "rows" spelling."""
        t0 = time.perf_counter()
        want = "rows" if want_logits and want is None else want
        eager = name not in self.traces
        if eager:
            # no capture yet (the serving warm-up's dummy decode runs before the trace phase): run the program eagerly
            outs = getattr(self, self._BODIES[name])()
            if want == "cands":
                want = "rows"
        else:
            ttnn.execute_trace(self.mesh, self.traces[name], cq_id=0, blocking=False)
            outs = self.outs[name]
            if want == "cands":
                ttnn.execute_trace(self.mesh, self.traces["cands_" + name], cq_id=0, blocking=False)
        res = None
        if outs:
            am, mx = outs[0], outs[1]
            comp = ttnn.ConcatMeshToTensor(self.mesh, dim=0)
            t1 = time.perf_counter()
            am_h = ttnn.to_torch(am, mesh_composer=comp).reshape(len(self.mesh.get_device_ids()), -1)  # [TP, B]
            mx_h = ttnn.to_torch(mx, mesh_composer=comp).float().reshape(am_h.shape[0], -1)
            self.timing["ids_read"].append(time.perf_counter() - t1)
            shard = mx_h.argmax(0)  # [B] winning shard per row
            width = self.V // am_h.shape[0]
            res = [int(am_h[shard[b], b].item()) + int(shard[b].item()) * width for b in range(am_h.shape[1])]
            if want == "rows" and len(outs) > 2:
                t1 = time.perf_counter()
                lg = ttnn.to_torch(outs[2], mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=-1))  # bf16 [1,1,B,V]
                self._last_logits = lg.reshape(-1, lg.shape[-1])[:, : self.V]  # [B, V] bf16 (float per used row)
                self.timing["lg_read"].append(time.perf_counter() - t1)
            elif want == "cands":
                t1 = time.perf_counter()
                self._last_cands = self._read_cands(self.outs["cands_" + name])
                self.timing["cands_read"].append(time.perf_counter() - t1)
        else:
            ttnn.synchronize_device(self.mesh)
        if eager:
            for t in outs:
                ttnn.deallocate(t)
        self.timing[name].append(time.perf_counter() - t0)
        return res

    def _write(self, dev_t, host_t):
        ttnn.copy_host_to_device_tensor(host_t, dev_t)

    def _read_cands(self, outs):
        """(values [B, TP*2*32] float, ids [B, TP*2*32] long with global vocab ids) from the candidate trace outputs
        (values / local ids per half, per shard)."""
        comp = ttnn.ConcatMeshToTensor(self.mesh, dim=0)
        TP = len(self.mesh.get_device_ids())
        width = self.V // TP
        vals, ids = [], []
        for hi in range(len(outs) // 2):
            v = ttnn.to_torch(outs[2 * hi], mesh_composer=comp).float()  # [TP, 1, B, 32]
            i = ttnn.to_torch(outs[2 * hi + 1], mesh_composer=comp).long()  # [TP, 1, B, 32] (uint16 / uint32 local)
            i = i + hi * self._cand_half + torch.arange(TP).view(TP, 1, 1, 1) * width
            vals.append(v)
            ids.append(i)
        v = torch.cat(vals, dim=-1)  # [TP, 1, B, n]
        i = torch.cat(ids, dim=-1)
        B = v.shape[2]
        v = v.permute(2, 0, 1, 3).reshape(B, -1)  # [B, TP*n]
        i = i.permute(2, 0, 1, 3).reshape(B, -1)
        return v, i

    def _target_fn(self, sampler, R):
        """Dense target distribution of row i from the pass's candidates (sparse requests)."""
        vals, ids = self._last_cands

        def P(i):
            return sampler.probs_topk(vals[i], ids[i], self.V)

        return P

    def draft(self, tok: int, pos: int, sampler=None):
        """k chained head steps from (tok, d_hid) at pos; returns the k drafts (and, with a non-greedy `sampler`, the
        draft distributions q_j the accept rule needs). d_hid must hold the hidden of `tok`."""
        drafts, qs = [], []
        sampling = sampler is not None and not sampler.greedy
        cur_tok, cur_pos = tok, pos
        S = self.d_tok.shape[-1]
        sparse = sampling and sampler.sparse_ok()
        for _ in range(self.k):
            self._write(self.d_tok, _host_u32([cur_tok], (1, 1, 1, S)))
            self._write(self.d_pos, _host_i32([cur_pos], (1,)))
            self._write(self.d_rot, _host_u32([cur_pos + self.rope_delta], (1, 1)))
            d = self._run("draft", want=("cands" if sparse else "rows") if sampling else None)[0]
            if sampling:
                if sparse and self._last_cands is not None:
                    vals, ids = self._last_cands
                    q = sampler.probs_topk(vals[0], ids[0], self.V)
                    self._last_cands = None
                else:
                    q = sampler.probs(self._last_logits[0])
                d = sampler.sample(q)
                qs.append(q)
            drafts.append(d)
            cur_tok, cur_pos = d, cur_pos + 1
        return (drafts, qs) if sampling else drafts

    def verify(self, rows, pos: int, want_logits=False, sampler=None):
        """Pass over rows (R = len(rows) <= k + 1) at positions pos..; returns the per-row argmax (R values). With
        `want_logits` also the target for SpecSampler.accept: the [R, V] logits rows, or (sparse `sampler`) a callable
        i -> dense target distribution built from the device top-k candidates."""
        R = len(rows)
        BW, S = self.BW, self.p_tok.shape[-1]
        positions = [pos + i for i in range(BW)]
        self._write(self.p_tok, _host_u32(list(rows) + [0] * (BW - R), (1, 1, 1, S)))
        self._write(self.p_pos, _host_i32(positions, (BW,)))
        self._write(self.p_rot, _host_u32([p + self.rope_delta for p in positions], (1, BW)))
        self._write(self.p_masks, rows_pos_masks_host(positions, BW))
        self.plen.stage(R)
        sparse = want_logits and sampler is not None and sampler.sparse_ok()
        self._last_cands = None
        preds = self._run("pass", want=("cands" if sparse else "rows") if want_logits else None)[:R]
        if not want_logits:
            return preds
        if sparse and self._last_cands is not None:
            return preds, self._target_fn(sampler, R)
        return preds, self._last_logits[:R]

    def commit(self, accepted_rows: int):
        self.plen_commit.stage(accepted_rows)
        self._run("commit")

    def head_rows(self, tokens, pos: int):
        """Head K/V for rows j = 0..: (token tokens[j], hidden h_out[j]) at position pos + 1 + j (tokens = rows[1..a])."""
        BW, S = self.BW, self.r_tok.shape[-1]
        positions = [pos + 1 + j for j in range(BW)]
        self._write(self.r_tok, _host_u32(list(tokens) + [0] * (BW - len(tokens)), (1, 1, 1, S)))
        self._write(self.r_pos, _host_i32(positions, (BW,)))
        self._write(self.r_rot, _host_u32([p + self.rope_delta for p in positions], (1, BW)))
        self._run("rows")

    def generate(self, tok0: int, pos: int, max_new_tokens: int, *, log=None):
        """Loop from the first sampled token tok0 at position pos with d_hid = the hidden that produced it (the head's
        prompt rows already written). Returns (tokens, passes)."""
        out_tokens = [tok0]
        passes = 0
        hist = [0] * (self.k + 1)
        while len(out_tokens) < max_new_tokens:
            t_cur = out_tokens[-1]
            drafts = self.draft(t_cur, pos)
            rows = [t_cur] + drafts
            preds = self.verify(rows, pos)
            a = 0
            for d, p in zip(drafts, preds):
                if d != p:
                    break
                a += 1
            nxt = preds[a]
            if a < self.k:
                self.commit(a + 1)
            if a > 0:
                self.head_rows(rows[1 : a + 1], pos)
            self._select_hidden(a)
            out_tokens.extend(rows[1 : a + 1])
            out_tokens.append(nxt)
            pos += a + 1
            passes += 1
            hist[a] += 1
            if log:
                log(f"pass {passes}: drafts {drafts} accepted {a} -> +{a + 1} tokens (total {len(out_tokens)})")
        return out_tokens[:max_new_tokens], passes, hist

    def timing_summary(self):
        return {k: (len(v), sum(v) / len(v) * 1e3 if v else 0.0) for k, v in self.timing.items()}
