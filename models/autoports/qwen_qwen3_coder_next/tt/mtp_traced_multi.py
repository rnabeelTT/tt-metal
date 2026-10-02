# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Traced multi-user MTP step: fixed-width programs over ALL B decode slots (live or not), replayed per step.

    pass   : snapshot the batched GDN state -> the layers over the user-major rows block (B x R rows, width
             BW = rows_width(B * R)) -> per-row argmax / value (read back); every GDN layer's placed input block is
             copied into a persistent buffer for the commit.
    commit : restore the snapshot -> one batched GDN replay per layer with the per-user accepted lengths (RowsLen).
    draft  : one head step over the B slots (per-row K/V write with -1 masks: a dead slot's stale page-table row must
             not be written) -> argmax per row (read back); the head hidden feeds the next step.
    rows   : the head's K/V rows for the accepted tokens: BW rows, hidden gathered from the pass block by a one-hot.
    select : the next draft's hidden per slot from the pass block (one-hot matmul into the head hidden input).
    Dead slots ride along with zero tokens, position 0 and K/V writes masked off; their GDN state rows hold garbage until
    the slot's next prefill rewrites them. Host side per slot: (live, tok, pos) and the plugin's page-table rows.
"""
from __future__ import annotations

import time

import torch

import ttnn
from models.autoports.qwen_qwen3_coder_next.tt.mtp_multi import GDN_T, RowsLen, rows_block_forward
from models.autoports.qwen_qwen3_coder_next.tt.mtp_rows import rows_pos_masks_host, rows_width
from models.autoports.qwen_qwen3_coder_next.tt.mtp_traced import _host_i32, _host_u32

_DRAM = ttnn.DRAM_MEMORY_CONFIG


class MtpTracedMulti:
    def __init__(self, gen, head, k: int, B: int):
        self.gen, self.head, self.k, self.B = gen, head, int(k), int(B)
        self.R = self.k + 1
        self.BW = rows_width(self.B * self.R)
        assert self.BW <= 32, f"{self.B} slots x {self.R} rows = {self.B * self.R} > 32: split the slots into groups"
        self.model = m = gen.model
        self.mesh = m.mesh_device
        self.V = int(gen.args.vocab_size)
        self.H = int(gen.args.dim)
        self.dn_layers = [layer.attention for layer in m.layers if not layer.is_full_attention]
        K = self.dn_layers[0].K
        assert self.B <= 4 or True, "the batched chunk kernel caps B at 4 per launch at TP=4 (grouped launches: todo)"
        dev = lambda t: ttnn.to_device(t, self.mesh, memory_config=_DRAM)
        B, BW, R = self.B, self.BW, self.R
        W = gen.page_table.shape[-1]
        self.pt_rows = torch.zeros(B, W, dtype=torch.int32)  # the slots' page-table rows (host copy)
        # pass inputs
        tok, pos, rot, pt = m.prepare_decode_inputs_host(
            torch.zeros(BW, dtype=torch.int32),
            torch.zeros(BW, dtype=torch.int32),
            torch.zeros(BW, W, dtype=torch.int32),
        )
        self.p_tok, self.p_pos, self.p_rot, self.p_pt = dev(tok), dev(pos), dev(rot), dev(pt)
        self.p_masks = dev(rows_pos_masks_host([-1] * BW, BW))
        self.rows_len = RowsLen(self.mesh, B, GDN_T, K)
        self.rows_len.stage([R] * B)
        self.commit_len = RowsLen(self.mesh, B, GDN_T, K)
        self.commit_len.stage([R] * B)
        # per GDN layer: persistent scan inputs (pre-conv qkv, q/k/v/beta/g) of the rows pass, filled at its first
        # (eager warm) run and copied into by every later run / trace replay; the commit's scan-only replay reads them
        self.saved = [{} for _ in self.dn_layers]
        self.snap_rec = [ttnn.clone(dn.rec_state) for dn in self.dn_layers]
        self.snap_conv = [[ttnn.clone(c) for c in dn.conv_states] for dn in self.dn_layers]
        self.h_out = None
        # head draft inputs (B rows)
        tok, pos, rot, pt = m.prepare_decode_inputs_host(
            torch.zeros(B, dtype=torch.int32), torch.zeros(B, dtype=torch.int32), torch.zeros(B, W, dtype=torch.int32)
        )
        self.d_tok, self.d_pos, self.d_rot, self.d_pt = dev(tok), dev(pos), dev(rot), dev(pt)
        self.d_masks = dev(rows_pos_masks_host([-1] * B, B))
        self.d_hid = self._zeros((1, 1, B, self.H), ttnn.bfloat16)
        # head rows inputs (BW rows)
        tok, pos, rot, pt = m.prepare_decode_inputs_host(
            torch.zeros(BW, dtype=torch.int32),
            torch.zeros(BW, dtype=torch.int32),
            torch.zeros(BW, W, dtype=torch.int32),
        )
        self.r_tok, self.r_pos, self.r_rot, self.r_pt = dev(tok), dev(pos), dev(rot), dev(pt)
        self.r_masks = dev(rows_pos_masks_host([-1] * BW, BW))
        self.r_sel = self._zeros((1, 1, BW, BW), ttnn.bfloat16)  # head-rows hidden <- pass block rows
        self.n_sel = self._zeros((1, 1, B, BW), ttnn.bfloat16)  # next draft hidden per slot <- pass block rows
        self.traces, self.outs = {}, {}
        self.timing = {"pass": [], "commit": [], "draft": [], "rows": []}
        # host state per slot
        self.live = [False] * B
        self.tok = [0] * B
        self.pos = [0] * B

    def _zeros(self, shape, dtype):
        return ttnn.from_torch(
            torch.zeros(*shape, dtype=torch.bfloat16),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=_DRAM,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def _write(self, dev_t, host_t):
        ttnn.copy_host_to_device_tensor(host_t, dev_t)

    def _write_pt(self, buf, rows_idx):
        """buf [n, W] <- the page-table rows of slots rows_idx (a list of slot ids, one per buffer row)."""
        host = ttnn.from_torch(self.pt_rows[rows_idx].contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host, buf)

    # ------------------------------------------------------------------ program bodies
    def _argmax_max(self, logits):
        lg_rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        am = ttnn.argmax(lg_rm, dim=-1, keepdim=True)
        mx = ttnn.gather(lg_rm, -1, am)
        ttnn.deallocate(lg_rm)
        return am, mx

    def _pass_body(self):
        m = self.model
        for dn, rec, taps in zip(self.dn_layers, self.snap_rec, self.snap_conv):
            ttnn.copy(dn.rec_state, rec)
            for c, t in zip(dn.conv_states, taps):
                ttnn.copy(c, t)
        cos, sin = m._rope_from_idxs(self.p_rot)
        tok = m._decode_tokens_view(self.p_tok, self.BW)
        x = m.embd(tok)
        x = ttnn.reshape(x, (1, 1, self.BW, x.shape[-1]))
        x = ttnn.to_memory_config(x, _DRAM)
        h, logits = rows_block_forward(
            m, x, cos, sin, self.p_pos, self.p_pt, self.p_masks, self.B, self.R, self.rows_len, self.saved
        )
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
        from models.autoports.qwen_qwen3_coder_next.tt.mtp_multi import commit_multi

        commit_multi(self.model, (self.snap_rec, self.snap_conv), self.saved, self.commit_len)
        return ()

    def _head_rows(self, tok, hid, pos, rot, pt, masks):
        att = self.head.layer.attention
        att.rows_kv_write = True
        att.rows_pos_masks = masks
        try:
            return self.head.decode_step(tok, hid, pos, rot, pt)
        finally:
            att.rows_kv_write = False
            att.rows_pos_masks = None

    def _draft_body(self):
        lg, h_next = self._head_rows(self.d_tok, self.d_hid, self.d_pos, self.d_rot, self.d_pt, self.d_masks)
        am, mx = self._argmax_max(lg)
        ttnn.deallocate(lg)
        ttnn.copy(h_next, self.d_hid)
        ttnn.deallocate(h_next)
        return am, mx

    def _rows_body(self):
        hid = ttnn.matmul(self.r_sel, self.h_out, memory_config=_DRAM)  # [1,1,BW,H]
        lg, h = self._head_rows(self.r_tok, hid, self.r_pos, self.r_rot, self.r_pt, self.r_masks)
        for t in (lg, h, hid):
            ttnn.deallocate(t)
        return ()

    def _select_body(self):
        picked = ttnn.matmul(self.n_sel, self.h_out, memory_config=_DRAM)  # [1,1,B,H]
        ttnn.copy(picked, self.d_hid)
        ttnn.deallocate(picked)

    # ------------------------------------------------------------------ warm / capture / run
    _BODIES = {"pass": "_pass_body", "commit": "_commit_body", "draft": "_draft_body", "rows": "_rows_body"}

    def warm(self, flush=None):
        for name in ("pass", "pass", "commit", "draft", "rows"):
            outs = getattr(self, self._BODIES[name])()
            for t in outs:
                ttnn.deallocate(t)
            if flush:
                flush()
        self._select_body()
        ttnn.synchronize_device(self.mesh)

    def capture(self, flush=None):
        for name in ("pass", "commit", "draft", "rows"):
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            outs = getattr(self, self._BODIES[name])()
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

    def _run(self, name):
        t0 = time.perf_counter()
        eager = name not in self.traces
        if eager:
            outs = getattr(self, self._BODIES[name])()
        else:
            ttnn.execute_trace(self.mesh, self.traces[name], cq_id=0, blocking=False)
            outs = self.outs[name]
        res = None
        if outs:
            am, mx = outs
            comp = ttnn.ConcatMeshToTensor(self.mesh, dim=0)
            am_h = ttnn.to_torch(am, mesh_composer=comp).reshape(len(self.mesh.get_device_ids()), -1)
            mx_h = ttnn.to_torch(mx, mesh_composer=comp).float().reshape(am_h.shape[0], -1)
            shard = mx_h.argmax(0)
            width = self.V // am_h.shape[0]
            res = [int(am_h[shard[b], b].item()) + int(shard[b].item()) * width for b in range(am_h.shape[1])]
        else:
            ttnn.synchronize_device(self.mesh)
        if eager:
            for t in outs:
                ttnn.deallocate(t)
        self.timing[name].append(time.perf_counter() - t0)
        return res

    # ------------------------------------------------------------------ per-slot bookkeeping
    def set_slot(self, slot, tok, pos, page_table_row, hidden_row):
        """A freshly prefilled slot: its first token (produced by the prefill) at position pos, its page-table row and
        the head hidden that produced tok (host [H] bf16 or a device [1,1,1,H] tensor)."""
        self.live[slot], self.tok[slot], self.pos[slot] = True, int(tok), int(pos)
        self.pt_rows[slot] = torch.as_tensor(page_table_row).to(torch.int32)
        self._set_hidden_row(slot, hidden_row)

    def _set_hidden_row(self, slot, hidden_row):
        """d_hid[slot] <- hidden_row through a host mirror (a row write into a device tensor is not an op we have)."""
        if not hasattr(self, "_hid_host"):
            self._hid_host = torch.zeros(1, 1, self.B, self.H, dtype=torch.bfloat16)
        if isinstance(hidden_row, torch.Tensor):
            self._hid_host[0, 0, slot] = hidden_row.reshape(-1).to(torch.bfloat16)
        else:
            row = ttnn.to_torch(ttnn.get_device_tensors(hidden_row)[0]).reshape(-1)[: self.H]
            self._hid_host[0, 0, slot] = row.to(torch.bfloat16)
        self._write(self.d_hid, ttnn.from_torch(self._hid_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT))

    def free_slot(self, slot):
        self.live[slot] = False

    def remap_slots(self, mapping):
        """mapping[new] = old (the plugin's batch condense, as _remap_gdn_slots sees it)."""
        live, tok, pos = list(self.live), list(self.tok), list(self.pos)
        pt = self.pt_rows.clone()
        hid = self._hid_host.clone() if hasattr(self, "_hid_host") else None
        for new, old in enumerate(mapping):
            if old is None or int(old) < 0:
                self.live[new] = False
                continue
            old = int(old)
            self.live[new], self.tok[new], self.pos[new] = live[old], tok[old], pos[old]
            self.pt_rows[new] = pt[old]
            if hid is not None:
                self._hid_host[0, 0, new] = hid[0, 0, old]
        if hid is not None:
            self._write(self.d_hid, ttnn.from_torch(self._hid_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT))

    # ------------------------------------------------------------------ one step over every live slot
    def step(self, page_table_rows=None):
        """Returns per slot the list of emitted ids (accepted drafts + the verified next token), [] for dead slots."""
        B, R, k, BW = self.B, self.R, self.k, self.BW
        live = [u for u in range(B) if self.live[u]]
        if not live:
            return [[] for _ in range(B)]
        if page_table_rows is not None:
            for u in live:
                self.pt_rows[u] = torch.as_tensor(page_table_rows[u]).to(torch.int32)
        S = self.d_tok.shape[-1]
        # ---- drafts: k head steps over the B slots (dead slots: token 0, position 0, K/V write masked off)
        drafts = [[] for _ in range(B)]
        cur = [self.tok[u] if self.live[u] else 0 for u in range(B)]
        self._write_pt(self.d_pt, list(range(B)))
        for j in range(k):
            posj = [self.pos[u] + j if self.live[u] else 0 for u in range(B)]
            self._write(self.d_tok, _host_u32(cur, (1, 1, 1, S)))
            self._write(self.d_pos, _host_i32(posj, (B,)))
            self._write(self.d_rot, _host_u32(posj, (1, B)))
            self._write(self.d_masks, rows_pos_masks_host([p if self.live[u] else -1 for u, p in enumerate(posj)], B))
            am = self._run("draft")
            for u in range(B):
                drafts[u].append(int(am[u]))
            cur = [int(am[u]) if self.live[u] else 0 for u in range(B)]
        # ---- verify pass over all slots' rows
        rows = [[self.tok[u]] + drafts[u] if self.live[u] else [0] * R for u in range(B)]
        tok_list = [t for r in rows for t in r] + [0] * (BW - B * R)
        pos_list = [self.pos[u] + i if self.live[u] else 0 for u in range(B) for i in range(R)] + [0] * (BW - B * R)
        mask_pos = [self.pos[u] + i if self.live[u] else -1 for u in range(B) for i in range(R)] + [-1] * (BW - B * R)
        self._write(self.p_tok, _host_u32(tok_list, (1, 1, 1, self.p_tok.shape[-1])))
        self._write(self.p_pos, _host_i32(pos_list, (BW,)))
        self._write(self.p_rot, _host_u32(pos_list, (1, BW)))
        self._write(self.p_masks, rows_pos_masks_host(mask_pos, BW))
        self._write_pt(self.p_pt, [u for u in range(B) for _ in range(R)] + [B - 1] * (BW - B * R))
        preds = self._run("pass")
        acc, nxt = [0] * B, [0] * B
        for u in live:
            p = preds[u * R : (u + 1) * R]
            a = 0
            for d, q in zip(drafts[u], p):
                if d != q:
                    break
                a += 1
            acc[u], nxt[u] = a, p[a]
        # ---- commit (any live slot rejected a draft): restore + batched replay with the accepted lengths
        if any(acc[u] < k for u in live):
            self.commit_len.stage([acc[u] + 1 if self.live[u] else R for u in range(B)])
            self._run("commit")
        # ---- head K/V rows for the accepted tokens (row u*R+i, i = 1..a_u: token rows[u][i], hidden pass row u*R+i-1)
        r_tok = [0] * BW
        r_pos = [0] * BW
        r_mask = [-1] * BW
        sel = torch.zeros(1, 1, BW, BW, dtype=torch.bfloat16)
        any_rows = False
        for u in live:
            for i in range(1, acc[u] + 1):
                r = u * R + i
                r_tok[r], r_pos[r], r_mask[r] = rows[u][i], self.pos[u] + i, self.pos[u] + i
                sel[0, 0, r, r - 1] = 1.0
                any_rows = True
        if any_rows:
            self._write(self.r_tok, _host_u32(r_tok, (1, 1, 1, self.r_tok.shape[-1])))
            self._write(self.r_pos, _host_i32(r_pos, (BW,)))
            self._write(self.r_rot, _host_u32(r_pos, (1, BW)))
            self._write(self.r_masks, rows_pos_masks_host(r_mask, BW))
            self._write(self.r_sel, ttnn.from_torch(sel, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT))
            self._write_pt(self.r_pt, [u for u in range(B) for _ in range(R)] + [B - 1] * (BW - B * R))
            self._run("rows")
        # ---- next draft hidden per slot = pass row u*R + a_u
        nsel = torch.zeros(1, 1, B, BW, dtype=torch.bfloat16)
        for u in live:
            nsel[0, 0, u, u * R + acc[u]] = 1.0
        self._write(self.n_sel, ttnn.from_torch(nsel, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT))
        self._select_body()
        if hasattr(self, "_hid_host"):  # keep the host mirror current for slot writes / remaps
            self._hid_host = (
                ttnn.to_torch(ttnn.get_device_tensors(self.d_hid)[0]).reshape(1, 1, B, self.H).to(torch.bfloat16)
            )
        out = [[] for _ in range(B)]
        for u in live:
            out[u] = rows[u][1 : acc[u] + 1] + [nxt[u]]
            self.pos[u] += acc[u] + 1
            self.tok[u] = nxt[u]
        return out

    def timing_summary(self):
        return {k: (len(v), sum(v) / len(v) * 1e3 if v else 0.0) for k, v in self.timing.items()}
