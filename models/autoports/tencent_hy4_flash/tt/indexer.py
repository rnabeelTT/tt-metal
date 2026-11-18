# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4 compressed indexer (HF HYV4FlashCompressedIndexer) for the sparse attention layers.

Per token: key k_t = RMSNorm(W_tk x) (128). Every 4 tokens pool into one group key (fp32 gated softmax over the 4 slots
+ intra-group position embedding, then RMSNorm), roped (last 64 dims) at the group's first position, bf16. Per query:
q = RoPE(W_q q_resid) (16 heads x 128), w = W_hw x / sqrt(16 * 128); score(g) = sum_h w_h relu(q_h . key_g). A query at
position p keeps the 511 best complete groups g < p // 4 plus its own group (tt/attention.index_lists expands them).
Rows with p < 2048 attend densely (all of their 511-or-fewer visible groups are kept), so they skip the scoring.

Day 0: the token keys are read back and the groups pooled on the host (exactly the reference's fp32 torch ops), the
group keys mirrored into a device cache; scoring (indexer_score_dsa) and the top-512 (topk_large_indices) run on the
device and the group ids are read back for the index lists.
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
        # host pooling weights (fp32, as the reference keeps them)
        self.Wv = sd[p + "key_value_projection.weight"].float()
        self.Wg = sd[p + "key_gate_projection.weight"].float()
        self.ape = sd[p + "intra_group_ape"].float()  # [4, 128]
        self.kn_w = sd[p + "key_norm.weight"].float()
        self.eps = cfg.rms_eps
        self.hw_scale = (self.nh**-0.5) * (self.d**-0.5)
        # host state: token keys of the open group, every group key (bf16, roped) for the device mirror
        self.gcap = -(-(max_ctx // self.r) // 32) * 32 + 32  # + the spare tile
        self.n_tok = 0
        self.open_keys = torch.zeros(0, self.d, dtype=torch.bfloat16)
        self.n_groups = 0
        self.gkeys = upload(mesh, torch.zeros(1, 1, self.gcap, self.d, dtype=torch.bfloat16), ttnn.bfloat16,
                            ttnn.ROW_MAJOR_LAYOUT)
        self.iota = upload(mesh, torch.arange(self.gcap, dtype=torch.float32).reshape(1, 1, 1, -1), ttnn.float32)

    def reset(self):
        self.n_tok, self.n_groups = 0, 0
        self.open_keys = torch.zeros(0, self.d, dtype=torch.bfloat16)

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

    def _append_tokens(self, tk: torch.Tensor, p0: int):
        """Token keys of positions p0.. (in order, continuing the sequence) -> pool and mirror complete groups."""
        assert p0 == self.n_tok, (p0, self.n_tok)
        allk = torch.cat([self.open_keys, tk])
        n_complete = allk.shape[0] // self.r
        if n_complete:
            gk = self._pool(allk[: n_complete * self.r], self.n_groups)
            g_rm = upload(self.mesh, gk.reshape(1, 1, n_complete, self.d), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.experimental.slice_write(g_rm, self.gkeys, [0, 0, self.n_groups, 0],
                                          [1, 1, self.n_groups + n_complete, self.d], [1, 1, 1, 1])
            ttnn.deallocate(g_rm)
            self.n_groups += n_complete
        self.open_keys = allk[n_complete * self.r :]
        self.n_tok = p0 + tk.shape[0]

    # ------------------------------------------------------------------ forward
    def forward(self, x, q_resid, p0: int, T: int, mem):
        """Updates the group keys with these T tokens; returns selected group ids [T, 512] (numpy, sorted by score) for
        the rows that need them (rows with p < 2048 get none), or None when no row does."""
        cfg = self.cfg
        Tp = int(x.shape[-2])
        tk = self.tk_norm(ttnn.matmul(x, self.wtk, compute_kernel_config=self.ckc, memory_config=mem), mem)
        tk_host = to_torch0(tk).reshape(Tp, self.d)[:T].to(torch.bfloat16)
        ttnn.deallocate(tk)
        self._append_tokens(tk_host, p0)
        if p0 + T - 1 < cfg.idx_candidate_topk:
            return None
        # ---- scoring: q [1, 16, Tp, 128] roped, w [1, 1, Tp, 16]
        q = ttnn.matmul(q_resid, self.wq, compute_kernel_config=self.ckc, memory_config=mem)
        q = ttnn.permute(ttnn.reshape(q, (1, Tp, self.nh, self.d)), (0, 2, 1, 3), memory_config=mem)
        qp = ttnn.slice(q, (0, 0, 0, 0), (1, self.nh, Tp, self.d - self.rope_d), memory_config=mem)
        qr = ttnn.slice(q, (0, 0, 0, self.d - self.rope_d), (1, self.nh, Tp, self.d), memory_config=mem)
        ttnn.deallocate(q)
        pos = torch.arange(p0, p0 + Tp)
        cos, sin = self.rope.tables(pos, self.theta)
        qr = self.rope(qr, cos, sin, mem)
        q = ttnn.concat([qp, qr], dim=-1, memory_config=mem)
        ttnn.deallocate(qp)
        ttnn.deallocate(qr)
        w = ttnn.multiply(ttnn.matmul(x, self.whw, compute_kernel_config=self.ckc, memory_config=mem), self.hw_scale,
                          memory_config=mem)
        n_vis_max = (p0 + T - 1) // self.r  # visible complete groups of the last row
        kv_len = -(-n_vis_max // 32) * 32
        # one spare tile past kv_len: indexer_score_dsa wants a tile-aligned chunk_start < its kv_len, and with
        # chunk_start = kv_len every real key is causally visible to every row (visibility is masked below)
        k = ttnn.slice(self.gkeys, (0, 0, 0, 0), (1, 1, kv_len + 32, self.d))
        k = ttnn.to_layout(k, ttnn.TILE_LAYOUT, memory_config=mem)
        vis = torch.tensor([(p0 + s) // self.r for s in range(Tp)], dtype=torch.float32).reshape(1, 1, Tp, 1)
        vis_t = upload(self.mesh, vis, ttnn.float32)
        iota = ttnn.slice(self.iota, (0, 0, 0, 0), (1, 1, 1, kv_len + 32))
        out = np.full((T, TOPK), 0xFFFFFFFF, dtype=np.uint32)
        for r0 in range(0, Tp, SCORE_ROWS):
            r1 = min(Tp, r0 + SCORE_ROWS)
            if r0 >= T:
                break
            qs = ttnn.slice(q, (0, 0, r0, 0), (1, self.nh, r1, self.d), memory_config=mem) if Tp > SCORE_ROWS else q
            ws = ttnn.slice(w, (0, 0, r0, 0), (1, 1, r1, self.nh), memory_config=mem) if Tp > SCORE_ROWS else w
            sc = ttnn.experimental.indexer_score_dsa(qs, k, ws, chunk_start_idx=kv_len, kv_len=kv_len + 32)
            sc = ttnn.typecast(sc, ttnn.float32, memory_config=mem)
            vs = ttnn.slice(vis_t, (0, 0, r0, 0), (1, 1, r1, 1), memory_config=mem)
            hide = ttnn.ge(iota, vs, memory_config=mem)  # [1,1,rows,kv_len]: group not visible to the row
            sc = ttnn.where(hide, float("-inf"), sc, memory_config=mem)
            sc = ttnn.typecast(sc, ttnn.bfloat16, memory_config=mem)
            sc_rm = ttnn.to_layout(sc, ttnn.ROW_MAJOR_LAYOUT, memory_config=mem)
            k_eff = TOPK if kv_len >= TOPK else max(16, (kv_len // 16) * 16)
            ids = ttnn.experimental.topk_large_indices(sc_rm, k=k_eff)
            ids_h = to_torch0(ids).reshape(r1 - r0, -1).numpy().view(np.uint32) if ids.dtype != ttnn.uint32 else \
                ttnn.to_torch(ttnn.get_device_tensors(ids)[0]).reshape(r1 - r0, -1).numpy().astype(np.int64).astype(np.uint32)
            n = min(T, r1) - r0
            out[r0 : r0 + n, : ids_h.shape[1]] = ids_h[:n]
            for t_ in (sc, sc_rm, ids, hide):
                ttnn.deallocate(t_)
        ttnn.deallocate(q)
        ttnn.deallocate(w)
        ttnn.deallocate(k)
        return out
