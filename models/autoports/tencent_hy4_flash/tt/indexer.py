# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4 compressed indexer (HF HYV4FlashCompressedIndexer) for the sparse attention layers.

Per token: key k_t = RMSNorm(W_tk x) (128). Every 4 tokens pool into one group key (fp32 gated softmax over the 4 slots
+ intra-group position embedding, then RMSNorm), roped (last 64 dims) at the group's first position, bf16. Per query:
q = RoPE(W_q q_resid) (16 heads x 128), w = W_hw x / sqrt(16 * 128); score(g) = sum_h w_h relu(q_h . key_g). A query at
position p keeps the 511 best complete groups g < p // 4 plus its own group (tt/attention.index_lists expands them).
Rows with p < 2048 attend densely (all of their 511-or-fewer visible groups are kept), so they skip the scoring.

The group keys live in a TILE device cache [1, 1, gcap, 128] (indexer_score_dsa's k).
- Prefill (eager, `forward`): token keys are read back and the groups pooled on the host (exactly the reference's
  fp32 torch ops); the host keeps a mirror of every group key and writes whole 32-group slabs of it.
- Decode (trace-safe, `step`): a 64-row token-key staging window and a 64-group group staging window, updated by
  exact one-hot matmuls the host supplies; the window's 16 groups are pooled on the device (fp32) every step and two
  32-group slabs are written. Scoring runs against a context bucket (kv_len fixed per trace) with the visibility of each
  row masked on the device; the top-512 groups expand to token indices on the device.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import ttnn

from models.autoports.tencent_hy4_flash.tt.common import RMSNorm, ckc, to_torch0, upload
from models.autoports.tencent_hy4_flash.tt.rope import DeviceRope, cos_sin, rope_torch

TOPK = 512
SCORE_ROWS = int(os.environ.get("HY4_IDX_SCORE_ROWS", "256"))


class Hy4Indexer:
    def __init__(self, mesh, cfg, sd, max_ctx: int):
        self.mesh, self.cfg = mesh, cfg
        self.nh, self.d, self.r = cfg.idx_heads, cfg.idx_dim, cfg.idx_ratio
        self.rope_d = cfg.qk_rope
        self.theta = cfg.rope_theta["sparse"]
        p = "compressed_indexer."
        bf = lambda t: t.to(torch.bfloat16)  # noqa: E731
        self.ckc = ckc(mesh, fp32=True)
        self.rope = DeviceRope(mesh, self.rope_d)
        self.wq = upload(mesh, bf(sd[p + "query_projection.weight"]).T.reshape(1, 1, cfg.q_lora, self.nh * self.d))
        self.whw = upload(mesh, bf(sd[p + "head_weight_projection.weight"]).T.reshape(1, 1, cfg.hidden, self.nh))
        self.wtk = upload(mesh, bf(sd[p + "token_key_projection.weight"]).T.reshape(1, 1, cfg.hidden, self.d))
        self.tk_norm = RMSNorm(mesh, sd[p + "token_key_norm.weight"], cfg.rms_eps)
        # pooling weights (fp32, as the reference keeps them): host copies (prefill) and device copies (decode)
        self.Wv = sd[p + "key_value_projection.weight"].float()
        self.Wg = sd[p + "key_gate_projection.weight"].float()
        self.ape = sd[p + "intra_group_ape"].float()  # [4, 128]
        self.kn_w = sd[p + "key_norm.weight"].float()
        self.eps = cfg.rms_eps
        self.hw_scale = (self.nh**-0.5) * (self.d**-0.5)
        self.dWvT = upload(mesh, self.Wv.T.contiguous().reshape(1, 1, self.d, self.d), ttnn.float32)
        self.dWgT = upload(mesh, self.Wg.T.contiguous().reshape(1, 1, self.d, self.d), ttnn.float32)
        self.dape = [upload(mesh, self.ape[j].reshape(1, 1, 1, self.d), ttnn.float32) for j in range(self.r)]
        self.dkn = upload(mesh, self.kn_w.reshape(1, 1, 1, self.d), ttnn.float32)
        # slot j of window group g' = token row r * g' + j: E_j [32, 64] (rows 16..31 zero)
        E = torch.zeros(self.r, 32, 64)
        for j in range(self.r):
            for g in range(64 // self.r):
                E[j, g, self.r * g + j] = 1.0
        self.E = [upload(mesh, E[j].reshape(1, 1, 32, 64), ttnn.float32) for j in range(self.r)]
        # group cache (TILE) + host mirror
        # + 64: the decode group window's two 32-group slabs past the context; + 32: the spare tile scoring reads
        self.gcap = -(-(max_ctx // self.r) // 32) * 32 + 96
        self.gkeys = upload(mesh, torch.zeros(1, 1, self.gcap, self.d, dtype=torch.bfloat16), ttnn.bfloat16)
        self._mirror = torch.zeros(self.gcap, self.d, dtype=torch.bfloat16)
        self._mirror_stale = None  # (g0, g1): device groups newer than the host mirror (device-pooled prefill)
        self._tk_pending = self._open_pending = None
        self._off_cache = {}
        self.iota = upload(mesh, torch.arange(self.gcap, dtype=torch.float32).reshape(1, 1, 1, -1), ttnn.float32)
        self.offs4 = upload(mesh, torch.arange(self.r, dtype=torch.float32).reshape(1, 1, 1, self.r), ttnn.float32)
        self.tk_stage = None
        self.g_stage = None
        self.reset()

    # host views of the device-pooled prefill's state, materialized on first use (decode's init_stage, a host-pooled
    # chunk after a device-pooled one): the device path itself never reads back
    @property
    def mirror(self):
        if self._mirror_stale is not None:
            g0, g1 = self._mirror_stale
            self._mirror_stale = None
            if g1 > g0:
                t = ttnn.slice(self.gkeys, (0, 0, g0, 0), (1, 1, g1, self.d))
                self._mirror[g0:g1] = to_torch0(t).reshape(g1 - g0, self.d).to(torch.bfloat16)
                ttnn.deallocate(t)
        return self._mirror

    @mirror.setter
    def mirror(self, v):
        self._mirror = v

    def _materialize(self, pending):
        t, n = pending
        return to_torch0(t).reshape(-1, self.d)[:n].to(torch.bfloat16)

    @property
    def tk_hist(self):
        if self._tk_pending is not None:
            self._tk_hist = self._materialize(self._tk_pending)
            self._tk_pending = None
        return self._tk_hist

    @tk_hist.setter
    def tk_hist(self, v):
        self._tk_hist, self._tk_pending = v, None

    @property
    def open_keys(self):
        if self._open_pending is not None:
            self._open_keys = self._materialize(self._open_pending)
            self._open_pending = None
        return self._open_keys

    @open_keys.setter
    def open_keys(self, v):
        self._open_keys, self._open_pending = v, None

    def reset(self):
        self._mirror_stale = None
        self.n_tok, self.n_groups = 0, 0
        self.open_keys = torch.zeros(0, self.d, dtype=torch.bfloat16)
        self.tk_hist = torch.zeros(0, self.d, dtype=torch.bfloat16)  # the last token keys (staging init)
        self.tk_hist_base = 0

    # ------------------------------------------------------------------ host pooling (reference-exact)
    def _pool(self, tk: torch.Tensor, g0: int) -> torch.Tensor:
        """tk [G * 4, 128] bf16 token keys of groups g0.. -> roped bf16 group keys [G, 128]."""
        G = tk.shape[0] // self.r
        grouped = tk.reshape(G, self.r, self.d).float()
        values = grouped @ self.Wv.T
        gates = grouped @ self.Wg.T + self.ape
        pooled = (values * gates.softmax(dim=1)).sum(dim=1)
        var = pooled.pow(2).mean(-1, keepdim=True)
        keys = (self.kn_w * (pooled * torch.rsqrt(var + self.eps))).to(torch.bfloat16)
        pos = torch.arange(g0, g0 + G) * self.r
        cos, sin = cos_sin(pos, self.theta, self.rope_d)
        rot = rope_torch(keys[:, self.d - self.rope_d :], cos, sin)
        return torch.cat([keys[:, : self.d - self.rope_d], rot], dim=-1)

    def _write_mirror(self, g0: int, g1: int):
        """Device group cache <- host mirror for groups [g0, g1) (whole 32-group slabs)."""
        _ = self.mirror  # sync any device-newer groups first (the slabs rewrite whole 32-group blocks)
        a, b = (g0 // 32) * 32, -(-g1 // 32) * 32
        for s0 in range(a, b, 32):  # 32-group slabs: the op wants the cache length a multiple of the written chunk
            slab = upload(self.mesh, self.mirror[s0 : s0 + 32].reshape(1, 1, 32, self.d), ttnn.bfloat16)
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(self.gkeys, slab, 0, 0, 1, s0, 0)
            ttnn.deallocate(slab)

    def _append_tokens(self, tk: torch.Tensor, p0: int):
        """Token keys of positions p0.. (in order, continuing the sequence) -> pool and mirror complete groups."""
        assert p0 == self.n_tok, (p0, self.n_tok)
        allk = torch.cat([self.open_keys, tk])
        n_complete = allk.shape[0] // self.r
        if n_complete:
            self.mirror[self.n_groups : self.n_groups + n_complete] = self._pool(allk[: n_complete * self.r],
                                                                                 self.n_groups)
            self._write_mirror(self.n_groups, self.n_groups + n_complete)
            self.n_groups += n_complete
        self.open_keys = allk[n_complete * self.r :]
        self.n_tok = p0 + tk.shape[0]
        hist = torch.cat([self.tk_hist, tk])
        keep = min(hist.shape[0], 128)
        self.tk_hist = hist[-keep:]
        self.tk_hist_base = self.n_tok - keep

    def _off(self, v):
        t = self._off_cache.get(v)
        if t is None:
            t = self._off_cache[v] = upload(self.mesh, torch.tensor([v], dtype=torch.int32).reshape(1, 1, 1, 1),
                                            ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        return t

    def _pool_prefill_device(self, tk, p0, T, Tp, mem):
        """The chunk's groups pooled on the device (decode's fp32 math) and written to the group cache: no readback.
        tk [1, 1, Tp, 128] bf16 token keys; p0 a multiple of 128 tokens, Tp of 128. Groups past the real tokens are
        junk, never visible (the rows' visibility) and rewritten once complete."""
        r, d = self.r, self.d
        G = Tp // r
        rm = ttnn.to_layout(tk, ttnn.ROW_MAJOR_LAYOUT, memory_config=mem)
        rows = ttnn.reshape(rm, (1, 1, G, r * d))  # row g = the 4 token keys of group g side by side
        K4 = ttnn.typecast(ttnn.to_layout(rows, ttnn.TILE_LAYOUT, memory_config=mem), ttnn.float32, memory_config=mem)
        ttnn.deallocate(rm)
        vals, gates = [], []
        for j in range(r):
            S = ttnn.slice(K4, (0, 0, 0, j * d), (1, 1, G, (j + 1) * d), memory_config=mem)
            vals.append(ttnn.matmul(S, self.dWvT, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=mem))
            g = ttnn.matmul(S, self.dWgT, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=mem)
            gates.append(ttnn.add(g, self.dape[j], memory_config=mem))
            ttnn.deallocate(S)
            ttnn.deallocate(g)
        ttnn.deallocate(K4)
        m = ttnn.maximum(ttnn.maximum(gates[0], gates[1], memory_config=mem),
                         ttnn.maximum(gates[2], gates[3], memory_config=mem), memory_config=mem)
        num = den = None
        for j in range(r):
            e = ttnn.exp(ttnn.subtract(gates[j], m, memory_config=mem), memory_config=mem)
            ev = ttnn.multiply(e, vals[j], memory_config=mem)
            num = ev if num is None else ttnn.add(num, ev, memory_config=mem)
            den = e if den is None else ttnn.add(den, e, memory_config=mem)
        for t_ in gates + vals + [m]:
            ttnn.deallocate(t_)
        pooled = ttnn.div(num, den, memory_config=mem)
        ttnn.deallocate(num)
        ttnn.deallocate(den)
        n = ttnn.rms_norm(pooled, epsilon=self.eps, compute_kernel_config=self.ckc, memory_config=mem)
        keys = ttnn.typecast(ttnn.multiply(n, self.dkn, memory_config=mem), ttnn.bfloat16, memory_config=mem)
        ttnn.deallocate(pooled)
        ttnn.deallocate(n)
        kp = ttnn.slice(keys, (0, 0, 0, 0), (1, 1, G, d - self.rope_d), memory_config=mem)
        kr = ttnn.slice(keys, (0, 0, 0, d - self.rope_d), (1, 1, G, d), memory_config=mem)
        ttnn.deallocate(keys)
        gcos, gsin = self.rope.tables(torch.arange(p0, p0 + Tp, r), self.theta)
        kr2 = self.rope(kr, gcos, gsin, mem)
        for t_ in (kr, gcos, gsin):
            ttnn.deallocate(t_)
        gk = ttnn.concat([kp, kr2], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(kp)
        ttnn.deallocate(kr2)
        g0 = p0 // r
        for s0 in range(0, G, 32):
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                self.gkeys, gk, self._off(0), self._off(g0 + s0), 0, 1, 0, input_row_offset=s0, input_rows=32)
        ttnn.deallocate(gk)
        # host bookkeeping (lazy views of the device state)
        n_end = p0 + T
        n_groups = n_end // r
        prev = self._mirror_stale
        self._mirror_stale = (prev[0] if prev else g0, n_groups)
        self.n_tok, self.n_groups = n_end, n_groups
        n_open = n_end - n_groups * r
        keep = min(T, 128)
        self._tk_pending = (ttnn.slice(tk, (0, 0, T - keep, 0), (1, 1, T, d)), keep)
        self.tk_hist_base = n_end - keep
        self._open_keys = torch.zeros(0, d, dtype=torch.bfloat16)
        self._open_pending = (ttnn.slice(tk, (0, 0, T - n_open, 0), (1, 1, T, d)), n_open) if n_open else None

    # ------------------------------------------------------------------ scoring (shared)
    def _query(self, x, q_resid, cos, sin, mem):
        """-> roped index query [1, 16, Tp, 128] and head weights [1, 1, Tp, 16]."""
        Tp = int(x.shape[-2])
        q = ttnn.matmul(q_resid, self.wq, compute_kernel_config=self.ckc, memory_config=mem)
        q = ttnn.permute(ttnn.reshape(q, (1, Tp, self.nh, self.d)), (0, 2, 1, 3), memory_config=mem)
        qp = ttnn.slice(q, (0, 0, 0, 0), (1, self.nh, Tp, self.d - self.rope_d), memory_config=mem)
        qr = ttnn.slice(q, (0, 0, 0, self.d - self.rope_d), (1, self.nh, Tp, self.d), memory_config=mem)
        ttnn.deallocate(q)
        qr2 = self.rope(qr, cos, sin, mem)
        ttnn.deallocate(qr)
        q = ttnn.concat([qp, qr2], dim=-1, memory_config=mem)
        ttnn.deallocate(qp)
        ttnn.deallocate(qr2)
        w = ttnn.multiply(ttnn.matmul(x, self.whw, compute_kernel_config=self.ckc, memory_config=mem), self.hw_scale,
                          memory_config=mem)
        return q, w

    def _topk(self, q, w, kv_len, vis, mem):
        """Masked scores against the first kv_len groups -> top-512 group ids [1, 1, rows, 512] uint32 RM."""
        sc = ttnn.experimental.indexer_score_dsa(q, self.gkeys, w, chunk_start_idx=kv_len, kv_len=kv_len + 32)
        rows = int(sc.shape[-2])
        if int(sc.shape[-1]) != kv_len + 32:  # the op's output spans the whole cache; only [0, kv_len) is written
            sc2 = ttnn.slice(sc, (0, 0, 0, 0), (1, 1, rows, kv_len + 32), memory_config=mem)
            ttnn.deallocate(sc)
            sc = sc2
        sc32 = ttnn.typecast(sc, ttnn.float32, memory_config=mem)
        ttnn.deallocate(sc)
        iota1 = ttnn.slice(self.iota, (0, 0, 0, 0), (1, 1, 1, kv_len + 32))
        iota = ttnn.repeat(iota1, ttnn.Shape((1, 1, rows, 1)), memory_config=mem)  # (no row x col broadcast in one op)
        ttnn.deallocate(iota1)
        vfull = ttnn.repeat(vis, ttnn.Shape((1, 1, 1, kv_len + 32)), memory_config=mem)  # same-shape compare
        hide = ttnn.ge(iota, vfull, memory_config=mem)  # [1, 1, rows, kv_len + 32]: not visible to the row
        ttnn.deallocate(vfull)
        m = ttnn.where(hide, float("-inf"), sc32, memory_config=mem)
        for t_ in (sc32, iota, hide):
            ttnn.deallocate(t_)
        mb = ttnn.typecast(m, ttnn.bfloat16, memory_config=mem)
        ttnn.deallocate(m)
        rm = ttnn.to_layout(mb, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(mb)
        ids = ttnn.experimental.topk_large_indices(rm, k=TOPK)
        ttnn.deallocate(rm)
        return ids

    def _expand(self, ids, rows, mem):
        """top-512 group ids [1, 1, rows, 512] uint32 RM -> token rows of the first 511 groups [1, 1, rows, 2044] uint32
        RM (device; the same math as step())."""
        it = ttnn.to_layout(ids, ttnn.TILE_LAYOUT, memory_config=mem)
        f = ttnn.typecast(it, ttnn.float32, memory_config=mem)
        ttnn.deallocate(it)
        n_sel = self.cfg.idx_group_topk - 1
        f511 = ttnn.slice(f, (0, 0, 0, 0), (1, 1, rows, n_sel), memory_config=mem)
        ttnn.deallocate(f)
        e = ttnn.add(ttnn.reshape(ttnn.multiply(f511, float(self.r), memory_config=mem), (1, rows, n_sel, 1)),
                     self.offs4, memory_config=mem)
        ttnn.deallocate(f511)
        e = ttnn.reshape(e, (1, 1, rows, n_sel * self.r))
        u = ttnn.typecast(e, ttnn.uint32, memory_config=mem)
        ttnn.deallocate(e)
        urm = ttnn.to_layout(u, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(u)
        return urm

    # ------------------------------------------------------------------ eager (prefill)
    def forward(self, x, q_resid, p0: int, T: int, mem, tail=None):
        """Updates the group keys with these T tokens; returns selected group ids [T, 512] (numpy, sorted by score) for
        the rows that need them (rows with p < 2048 get none), or None when no row does."""
        cfg = self.cfg
        Tp = int(x.shape[-2])
        tk = self.tk_norm(ttnn.matmul(x, self.wtk, compute_kernel_config=self.ckc, memory_config=mem), mem)
        if (os.environ.get("HY4_PF_DEVICE_POOL", "1") == "1" and p0 == self.n_tok and p0 % self.r == 0
                and Tp % (32 * self.r) == 0 and (p0 // self.r) % 32 == 0
                and self._open_pending is None and self._open_keys.shape[0] == 0):
            self._pool_prefill_device(tk, p0, T, Tp, mem)
        else:
            tk_host = to_torch0(tk).reshape(Tp, self.d)[:T].to(torch.bfloat16)
            self._append_tokens(tk_host, p0)
        ttnn.deallocate(tk)
        if p0 + T - 1 < cfg.idx_candidate_topk:
            return None
        cos, sin = self.rope.tables(torch.arange(p0, p0 + Tp), self.theta)
        q, w = self._query(x, q_resid, cos, sin, mem)
        n_vis_max = (p0 + T - 1) // self.r  # visible complete groups of the last row
        kv_len = -(-n_vis_max // 32) * 32
        vis = torch.tensor([(p0 + min(s, T - 1)) // self.r for s in range(Tp)], dtype=torch.float32)
        vis_t = upload(self.mesh, vis.reshape(1, 1, Tp, 1), ttnn.float32)
        if tail is not None and p0 >= cfg.idx_candidate_topk:
            # every row selects: the index lists are built on the device (no top-k readback), with `tail`
            # [1, 1, Tp, K - 2044] (sink, own group, sentinels) shared by the sparse layers of the chunk
            parts = []
            for r0 in range(0, Tp, SCORE_ROWS):
                r1 = min(Tp, r0 + SCORE_ROWS)
                qs = ttnn.slice(q, (0, 0, r0, 0), (1, self.nh, r1, self.d), memory_config=mem)
                ws = ttnn.slice(w, (0, 0, r0, 0), (1, 1, r1, self.nh), memory_config=mem)
                vs = ttnn.slice(vis_t, (0, 0, r0, 0), (1, 1, r1, 1), memory_config=mem)
                ids = self._topk(qs, ws, kv_len, vs, mem)
                parts.append(self._expand(ids, r1 - r0, mem))
                for t_ in (qs, ws, vs, ids):
                    ttnn.deallocate(t_)
            sel = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if len(parts) > 1:
                for t_ in parts:
                    ttnn.deallocate(t_)
            idx = ttnn.concat([sel, tail], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(sel)
            for t_ in (q, w, vis_t, cos, sin):
                ttnn.deallocate(t_)
            return ("device", idx)
        out = np.full((T, TOPK), 0xFFFFFFFF, dtype=np.uint32)
        for r0 in range(0, min(Tp, -(-T // 32) * 32), SCORE_ROWS):
            r1 = min(Tp, r0 + SCORE_ROWS)
            qs = ttnn.slice(q, (0, 0, r0, 0), (1, self.nh, r1, self.d), memory_config=mem)
            ws = ttnn.slice(w, (0, 0, r0, 0), (1, 1, r1, self.nh), memory_config=mem)
            vs = ttnn.slice(vis_t, (0, 0, r0, 0), (1, 1, r1, 1), memory_config=mem)
            ids = self._topk(qs, ws, kv_len, vs, mem)
            ids_h = ttnn.to_torch(ttnn.get_device_tensors(ids)[0]).reshape(r1 - r0, -1).numpy().astype(np.int64)
            n = min(T, r1) - r0
            out[r0 : r0 + n] = ids_h[:n].astype(np.uint32)
            for t_ in (qs, ws, vs, ids):
                ttnn.deallocate(t_)
        for t_ in (q, w, vis_t, cos, sin):
            ttnn.deallocate(t_)
        return out

    # ------------------------------------------------------------------ trace-safe step (decode / verify rows)
    def init_stage(self, base: int, gbase: int):
        """After a prefill: token staging = token keys of positions [base, base + 64) (those written so far), group
        staging = group keys [gbase, gbase + 64)."""
        tk = torch.zeros(64, self.d, dtype=torch.bfloat16)
        for i in range(64):
            pos = base + i
            j = pos - self.tk_hist_base
            if pos < self.n_tok and 0 <= j < self.tk_hist.shape[0]:
                tk[i] = self.tk_hist[j]
        t = upload(self.mesh, tk.reshape(1, 1, 64, self.d), ttnn.bfloat16)
        g = upload(self.mesh, self.mirror[gbase : gbase + 64].reshape(1, 1, 64, self.d).contiguous(), ttnn.bfloat16)
        if self.tk_stage is None:
            self.tk_stage, self.g_stage = t, g
        else:
            ttnn.copy(t, self.tk_stage)
            ttnn.copy(g, self.g_stage)
            ttnn.deallocate(t)
            ttnn.deallocate(g)

    def _pool_device(self, mem, gcos, gsin):
        """The 16 groups of the token staging window -> roped bf16 group keys [1, 1, 32, 128] (rows 16..31 junk)."""
        K = ttnn.typecast(self.tk_stage, ttnn.float32, memory_config=mem)
        vals, gates = [], []
        for j in range(self.r):
            S = ttnn.matmul(self.E[j], K, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=mem)
            vals.append(ttnn.matmul(S, self.dWvT, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=mem))
            g = ttnn.matmul(S, self.dWgT, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=mem)
            gates.append(ttnn.add(g, self.dape[j], memory_config=mem))
            ttnn.deallocate(S)
            ttnn.deallocate(g)
        ttnn.deallocate(K)
        m = ttnn.maximum(ttnn.maximum(gates[0], gates[1], memory_config=mem),
                         ttnn.maximum(gates[2], gates[3], memory_config=mem), memory_config=mem)
        num = den = None
        for j in range(self.r):
            e = ttnn.exp(ttnn.subtract(gates[j], m, memory_config=mem), memory_config=mem)
            ev = ttnn.multiply(e, vals[j], memory_config=mem)
            num = ev if num is None else ttnn.add(num, ev, memory_config=mem)
            den = e if den is None else ttnn.add(den, e, memory_config=mem)
        for t_ in gates + vals + [m]:
            ttnn.deallocate(t_)
        pooled = ttnn.div(num, den, memory_config=mem)
        ttnn.deallocate(num)
        ttnn.deallocate(den)
        n = ttnn.rms_norm(pooled, epsilon=self.eps, compute_kernel_config=self.ckc, memory_config=mem)
        keys = ttnn.typecast(ttnn.multiply(n, self.dkn, memory_config=mem), ttnn.bfloat16, memory_config=mem)
        ttnn.deallocate(pooled)
        ttnn.deallocate(n)
        kp = ttnn.slice(keys, (0, 0, 0, 0), (1, 1, 32, self.d - self.rope_d), memory_config=mem)
        kr = ttnn.slice(keys, (0, 0, 0, self.d - self.rope_d), (1, 1, 32, self.d), memory_config=mem)
        ttnn.deallocate(keys)
        kr2 = self.rope(kr, gcos, gsin, mem)
        ttnn.deallocate(kr)
        out = ttnn.concat([kp, kr2], dim=-1, memory_config=mem)
        ttnn.deallocate(kp)
        ttnn.deallocate(kr2)
        return out

    def step(self, x, q_resid, si, mem):
        """Token / group staging updates and slab writes; in the sparse regime the index lists of the rows
        [1, 1, 32, K] uint32 RM (device-built), else si.idx_dense."""
        tk = self.tk_norm(ttnn.matmul(x, self.wtk, compute_kernel_config=self.ckc, memory_config=mem), mem)
        a = ttnn.matmul(si.A, self.tk_stage, compute_kernel_config=self.ckc, memory_config=mem)
        b = ttnn.matmul(si.P, tk, compute_kernel_config=self.ckc, memory_config=mem)
        ttnn.deallocate(tk)
        new = ttnn.add(a, b, memory_config=mem)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        ttnn.copy(new, self.tk_stage)
        ttnn.deallocate(new)
        gk = self._pool_device(mem, si.gcos, si.gsin)
        a = ttnn.matmul(si.Ag, self.g_stage, compute_kernel_config=self.ckc, memory_config=mem)
        b = ttnn.matmul(si.Pg, gk, compute_kernel_config=self.ckc, memory_config=mem)
        ttnn.deallocate(gk)
        new = ttnn.add(a, b, memory_config=mem)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        ttnn.copy(new, self.g_stage)
        ttnn.deallocate(new)
        for j in range(2):
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                self.gkeys, self.g_stage, si.slot, si.off_grp[j], 0, 1, 0, input_row_offset=32 * j, input_rows=32)
        if not si.sparse:
            return si.idx_dense
        cos, sin = si.cos_sin["sparse"]
        q, w = self._query(x, q_resid, cos, sin, mem)
        ids = self._topk(q, w, si.bucket, si.vis, mem)
        ttnn.deallocate(q)
        ttnn.deallocate(w)
        # group ids -> token rows: the first 511 groups x 4 (+ the tail: sink, own group up to p, sentinels)
        it = ttnn.to_layout(ids, ttnn.TILE_LAYOUT, memory_config=mem)
        ttnn.deallocate(ids)
        f = ttnn.typecast(it, ttnn.float32, memory_config=mem)
        ttnn.deallocate(it)
        n_sel = self.cfg.idx_group_topk - 1
        f511 = ttnn.slice(f, (0, 0, 0, 0), (1, 1, 32, n_sel), memory_config=mem)
        ttnn.deallocate(f)
        e = ttnn.add(ttnn.reshape(ttnn.multiply(f511, float(self.r), memory_config=mem), (1, 32, n_sel, 1)),
                     self.offs4, memory_config=mem)  # [1, 32, 511, 4]
        ttnn.deallocate(f511)
        e = ttnn.reshape(e, (1, 1, 32, n_sel * self.r))
        u = ttnn.typecast(e, ttnn.uint32, memory_config=mem)
        ttnn.deallocate(e)
        urm = ttnn.to_layout(u, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(u)
        idx = ttnn.concat([urm, si.idx_tail], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.sel = urm  # the rows' selected token rows [1, 1, 32, 2044] (kept: the DFly2 drafter's index share)
        return idx
