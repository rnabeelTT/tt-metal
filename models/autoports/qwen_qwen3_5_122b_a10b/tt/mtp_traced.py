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
from models.autoports.qwen_qwen3_5_122b_a10b.tt.mtp_rows import (
    GDN_CHUNK,
    layer_forward_rows,
    rebuild_conv_carry,
    rows_pos_masks_host,
    rows_width,
)
from models.autoports.qwen_qwen3_5_122b_a10b.tt.prefill_len import PrefillLen

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
        self.timing = {"pass": [], "commit": [], "draft": [], "rows": []}

    # ------------------------------------------------------------------------------------------------ programs
    def _argmax_max(self, logits):
        """Per-shard argmax + max over the vocab dim of [1,1,B,V/TP] (read back by the host, combined across shards)."""
        # argmax on a TILE input is single-core (9.2 ms for 8 x 62080!); the ROW_MAJOR path is multi-core
        lg_rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        am = ttnn.argmax(lg_rm, dim=-1, keepdim=True)  # [1,1,B,1] uint32
        if os.environ.get("QWEN35_MTP_ARGMAX_MODE", "gather") == "gather":
            # the winning value by a gather at the argmax index (ttnn.max over the tile logits cost 244 us)
            mx = ttnn.gather(lg_rm, -1, am)  # [1,1,B,1] (gather takes UINT32 / UINT16 indices; argmax gives UINT32)
        else:
            mx = ttnn.max(logits, dim=-1, keepdim=True)  # [1,1,B,1]
        ttnn.deallocate(lg_rm)
        return am, mx

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
        am, mx = self._argmax_max(logits)
        ttnn.deallocate(logits)
        for t in (cos, sin):
            ttnn.deallocate(t)
        return am, mx

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
        am, mx = self._argmax_max(lg)
        ttnn.deallocate(lg)
        ttnn.copy(h_next, self.d_hid)
        ttnn.deallocate(h_next)
        return am, mx

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
        ttnn.synchronize_device(self.mesh)

    def release(self):
        for tid in self.traces.values():
            ttnn.release_trace(self.mesh, tid)
        self.traces = {}

    # ------------------------------------------------------------------------------------------------ per-pass host side
    _BODIES = {"pass": "_pass_body", "commit": "_commit_body", "draft": "_draft_body", "rows": "_rows_body"}

    def _run(self, name):
        t0 = time.perf_counter()
        eager = name not in self.traces
        if eager:
            # no capture yet (the serving warm-up's dummy decode runs before the trace phase): run the program eagerly
            outs = getattr(self, self._BODIES[name])()
        else:
            ttnn.execute_trace(self.mesh, self.traces[name], cq_id=0, blocking=False)
            outs = self.outs[name]
        res = None
        if outs:
            am, mx = outs
            comp = ttnn.ConcatMeshToTensor(self.mesh, dim=0)
            am_h = ttnn.to_torch(am, mesh_composer=comp).reshape(len(self.mesh.get_device_ids()), -1)  # [TP, B]
            mx_h = ttnn.to_torch(mx, mesh_composer=comp).float().reshape(am_h.shape[0], -1)
            shard = mx_h.argmax(0)  # [B] winning shard per row
            width = self.V // am_h.shape[0]
            res = [int(am_h[shard[b], b].item()) + int(shard[b].item()) * width for b in range(am_h.shape[1])]
        else:
            ttnn.synchronize_device(self.mesh)
        if eager:
            for t in outs:
                ttnn.deallocate(t)
        self.timing[name].append(time.perf_counter() - t0)
        return res

    def _write(self, dev_t, host_t):
        ttnn.copy_host_to_device_tensor(host_t, dev_t)

    def draft(self, tok: int, pos: int):
        """k chained head steps from (tok, d_hid) at pos; returns the k drafts. d_hid must hold the hidden of `tok`."""
        drafts = []
        cur_tok, cur_pos = tok, pos
        S = self.d_tok.shape[-1]
        for _ in range(self.k):
            self._write(self.d_tok, _host_u32([cur_tok], (1, 1, 1, S)))
            self._write(self.d_pos, _host_i32([cur_pos], (1,)))
            self._write(self.d_rot, _host_u32([cur_pos], (1, 1)))
            d = self._run("draft")[0]
            drafts.append(d)
            cur_tok, cur_pos = d, cur_pos + 1
        return drafts

    def verify(self, rows, pos: int):
        """Pass over rows (R = len(rows) <= k + 1) at positions pos..; returns the per-row argmax (R values)."""
        R = len(rows)
        BW, S = self.BW, self.p_tok.shape[-1]
        positions = [pos + i for i in range(BW)]
        self._write(self.p_tok, _host_u32(list(rows) + [0] * (BW - R), (1, 1, 1, S)))
        self._write(self.p_pos, _host_i32(positions, (BW,)))
        self._write(self.p_rot, _host_u32(positions, (1, BW)))
        self._write(self.p_masks, rows_pos_masks_host(positions, BW))
        self.plen.stage(R)
        return self._run("pass")[:R]

    def commit(self, accepted_rows: int):
        self.plen_commit.stage(accepted_rows)
        self._run("commit")

    def head_rows(self, tokens, pos: int):
        """Head K/V for rows j = 0..: (token tokens[j], hidden h_out[j]) at position pos + 1 + j (tokens = rows[1..a])."""
        BW, S = self.BW, self.r_tok.shape[-1]
        positions = [pos + 1 + j for j in range(BW)]
        self._write(self.r_tok, _host_u32(list(tokens) + [0] * (BW - len(tokens)), (1, 1, 1, S)))
        self._write(self.r_pos, _host_i32(positions, (BW,)))
        self._write(self.r_rot, _host_u32(positions, (1, BW)))
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
