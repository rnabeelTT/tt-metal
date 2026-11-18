# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4 gated MLA with learned sinks (HF HYV4FlashAttention), sparse (indexed) and sliding-window kinds.

TP over the (1, N) mesh by heads (sparse: 32 -> 8 per die, SWA: 64 -> 16). Absorbed MLA: one cached latent row per
token, [c_kv (512, normed) | k_rope (64, roped) | sink columns (32)] bf16 ROW-MAJOR, replicated on every die:

    score(h, t) = scale * (q_nope_h . W_uk_h c_t + q_rope_h . k_rope_t) = scale * ([W_uk_h^T q_nope_h | q_rope_h] . row_t)
    out_h       = W_uv_h (sum_t p_ht c_t)                      (V = the leading 512 columns of the row)

`ttnn.transformer.sparse_sdpa` computes softmax over an explicit index list per query row (the sparse layers' indexer
selection, the window of the SWA layers, or every earlier token while dense). The learned sink of head h is one more
key with score s_h and value 0: a reserved cache row whose sink columns hold 16 (with scale 1/16 that is a unit key)
and zeros elsewhere, scored against q's sink columns [s_hi, s_lo] (s_h split into two bf16 parts, so the sink logit
keeps ~16 bits). Every index list starts with that row.

Then: out * sigmoid(gate_proj(x)) -> o_proj (row parallel) -> all-reduce -> post_proj_norm (replicated).

Day-0 execution is eager: the host builds each pass's index lists (positions are host knowledge), the device scores
the compressed indexer and takes its top-512 groups (read back), and the indexer's group keys are pooled on the host
in fp32 from the read-back token keys (bit-for-bit the reference's pooling). See tt/indexer.py.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import ttnn

from models.autoports.tencent_hy4_flash.tt.common import (
    AllReduce,
    RMSNorm,
    TILE,
    ckc,
    num_devices,
    upload,
)
from models.autoports.tencent_hy4_flash.tt.rope import DeviceRope

SKF = getattr(ttnn.transformer, "SparseKVFormat", None) or getattr(ttnn, "SparseKVFormat", None)
SENTINEL = 0xFFFFFFFF
K_CHUNK = 128  # sparse_sdpa key chunk; index lists are padded to a multiple of it
SINK_COLS = 32
SINK_KEY = 16.0


def _dt(name, default):
    return {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}[os.environ.get(name, default)]


class Hy4Attention:
    def __init__(self, mesh, cfg, sd, layer_idx: int, max_ctx: int, chunk: int):
        """sd: the layer's `self_attn.` tensors (prefix stripped). max_ctx: cache capacity (sparse layers); chunk: the
        largest rows per forward (prefill chunk), which sizes the SWA ring."""
        self.mesh, self.cfg, self.li = mesh, cfg, layer_idx
        self.kind = cfg.layer_kind[layer_idx]
        self.N = N = num_devices(mesh)
        self.H = H = cfg.heads[self.kind]
        assert H % N == 0
        self.Hd = Hd = H // N
        self.Hp = max(32, -(-Hd // 32) * 32)  # sparse_sdpa wants a multiple of 32 heads
        self.nope, self.rope_d, self.vd, self.kvl = cfg.qk_nope, cfg.qk_rope, cfg.v_head, cfg.kv_lora
        self.width = cfg.latent_width + SINK_COLS  # 608
        self.scale = (cfg.qk_nope + cfg.qk_rope) ** -0.5
        assert abs(self.scale * SINK_KEY - 1.0) < 1e-12, "the sink trick assumes scale = 1/16"
        self.theta = cfg.rope_theta[self.kind]
        self.ar = AllReduce(mesh)
        self.ckc = ckc(mesh, fp32=True)
        self.ckc_lofi = ckc(mesh, ttnn.MathFidelity.HiFi2, fp32=True)
        self.rope = DeviceRope(mesh, self.rope_d)
        wdt = _dt("HY4_ATTN_DTYPE", "bfp8")
        sdt = _dt("HY4_ATTN_SMALL_DTYPE", "bf16")
        bf = lambda t: t.to(torch.bfloat16)  # noqa: E731

        # q: q_a (replicated) -> q_a_layernorm -> q_b (column parallel by heads)
        self.wqa = upload(mesh, bf(sd["q_a_proj.weight"]).T.reshape(1, 1, cfg.hidden, cfg.q_lora), sdt)
        self.q_norm = RMSNorm(mesh, sd["q_a_layernorm.weight"], cfg.rms_eps)
        self.wqb = upload(mesh, bf(sd["q_b_proj.weight"]).T.reshape(1, 1, cfg.q_lora, H * 256), wdt, mapper=("shard", -1))
        # kv: kv_a (replicated) -> [c (512) | k_rope (64)]
        self.wkva = upload(mesh, bf(sd["kv_a_proj_with_mqa.weight"]).T.reshape(1, 1, cfg.hidden, cfg.latent_width), sdt)
        self.kv_norm = RMSNorm(mesh, sd["kv_a_layernorm.weight"], cfg.rms_eps)
        # absorbed kv_b: W_uk [H, 192, 512], W_uv^T [H, 512, 256] (sharded by heads)
        kvb = bf(sd["kv_b_proj.weight"]).reshape(H, self.nope + self.vd, self.kvl)
        wuk = kvb[:, : self.nope, :].contiguous()  # [H, 192, 512]
        wuvT = kvb[:, self.nope :, :].transpose(1, 2).contiguous()  # [H, 512, 256]
        self.wuk = upload(mesh, wuk.reshape(1, H, self.nope, self.kvl), sdt, mapper=("shard", 1))
        self.wuvT = upload(mesh, wuvT.reshape(1, H, self.kvl, self.vd), sdt, mapper=("shard", 1))
        # output gate (column parallel), o_proj (row parallel), post_proj_norm
        self.wg = upload(mesh, bf(sd["linear_gate.weight"]).T.reshape(1, 1, cfg.hidden, H * self.vd), wdt, mapper=("shard", -1))
        self.wo = upload(mesh, bf(sd["o_proj.weight"]).T.reshape(1, 1, H * self.vd, cfg.hidden), wdt, mapper=("shard", -2))
        self.post_norm = RMSNorm(mesh, sd["post_proj_norm.weight"], cfg.rms_eps)
        # sinks: q's sink columns per (local, padded) head: [s_hi, s_lo, 0, ...]
        s = sd["learnable_sink_param"].float().reshape(H)
        s_hi = s.to(torch.bfloat16).float()
        s_lo = (s - s_hi).to(torch.bfloat16).float()
        qs = torch.zeros(N, self.Hp, SINK_COLS)
        for d in range(N):
            qs[d, :Hd, 0] = s_hi[d * Hd : (d + 1) * Hd]
            qs[d, :Hd, 1] = s_lo[d * Hd : (d + 1) * Hd]
        self.q_sink = upload(mesh, qs.reshape(N, self.Hp, 1, SINK_COLS), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT,
                             mapper=("shard", 0))
        # caches (row-major bf16, replicated): sparse layers hold every position + a sink block at the end; SWA layers
        # a ring of `ring` rows + the sink block
        if self.kind == "sparse":
            self.rows = -(-max_ctx // 32) * 32
        else:
            self.rows = -(-(chunk + cfg.window + 32) // 32) * 32
        self.sink_row = self.rows
        cache = torch.zeros(1, 1, self.rows + 32, self.width, dtype=torch.bfloat16)
        cache[0, 0, self.sink_row, cfg.latent_width] = SINK_KEY
        cache[0, 0, self.sink_row, cfg.latent_width + 1] = SINK_KEY
        self.cache = upload(mesh, cache, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        del cache
        self.indexer = None
        if self.kind == "sparse":
            from models.autoports.tencent_hy4_flash.tt.indexer import Hy4Indexer

            self.indexer = Hy4Indexer(mesh, cfg, sd, max_ctx)

    # ------------------------------------------------------------------ host-side index lists
    def _cache_row(self, pos: np.ndarray) -> np.ndarray:
        return pos if self.kind == "sparse" else pos % self.rows

    def index_lists(self, p0: int, T: int, selected: np.ndarray | None) -> torch.Tensor:
        """[1, 1, Tp, K] uint32 per query row p = p0 + s: [sink | keys... | sentinels]. keys: every position <= p
        (dense: sparse layers with p < 2048), the window (p - 511 .. p) for SWA, or the indexer's 511 groups x 4 +
        the query's own group up to p (selected[s] = the row's group ids, sorted by score)."""
        cfg = self.cfg
        Tp = -(-T // 32) * 32
        if self.kind == "swa":
            K = -(-(cfg.window + 1) // K_CHUNK) * K_CHUNK
        else:
            K = -(-(cfg.idx_candidate_topk + 1) // K_CHUNK) * K_CHUNK
        out = np.full((Tp, K), SENTINEL, dtype=np.uint32)
        out[:, 0] = self.sink_row
        for s in range(T):
            p = p0 + s
            if self.kind == "swa":
                keys = np.arange(max(0, p - cfg.window + 1), p + 1)
            elif selected is None or p < cfg.idx_candidate_topk:
                keys = np.arange(0, p + 1)
            else:
                g = selected[s][: cfg.idx_group_topk - 1].astype(np.int64)
                own = (p // cfg.idx_ratio) * cfg.idx_ratio
                keys = np.concatenate(
                    [(g[:, None] * cfg.idx_ratio + np.arange(cfg.idx_ratio)[None, :]).reshape(-1), np.arange(own, p + 1)]
                )
            out[s, 1 : 1 + keys.size] = self._cache_row(keys)
        # padding rows (s >= T) attend to the sink only: a defined (zero) output
        return torch.from_numpy(out.view(np.int32)).reshape(1, 1, Tp, K)

    # ------------------------------------------------------------------ forward
    def forward(self, x, p0: int, T: int, cos_sin, mem=None):
        """x: [1, 1, Tp, 3072] bf16 replicated (normed attention input), rows p0 .. p0 + T - 1 real (Tp = T padded to
        32). cos_sin: {kind: (cos, sin)} fp32 [1, 1, Tp, 64] device tables of these positions (shared by the layers).
        Returns [1, 1, Tp, 3072] bf16 replicated."""
        cfg, mesh = self.cfg, self.mesh
        mem = mem or ttnn.DRAM_MEMORY_CONFIG
        Tp = int(x.shape[-2])
        cos, sin = cos_sin[self.kind]
        # ---- q
        qa = ttnn.matmul(x, self.wqa, compute_kernel_config=self.ckc, memory_config=mem)
        q_resid = self.q_norm(qa, mem)
        ttnn.deallocate(qa)
        q = ttnn.matmul(q_resid, self.wqb, compute_kernel_config=self.ckc, memory_config=mem)  # [1,1,Tp,Hd*256]
        q = ttnn.permute(ttnn.reshape(q, (1, Tp, self.Hd, 256)), (0, 2, 1, 3), memory_config=mem)  # [1,Hd,Tp,256]
        q_nope = ttnn.slice(q, (0, 0, 0, 0), (1, self.Hd, Tp, self.nope), memory_config=mem)
        q_rope = ttnn.slice(q, (0, 0, 0, self.nope), (1, self.Hd, Tp, 256), memory_config=mem)
        ttnn.deallocate(q)
        q_lat = ttnn.matmul(q_nope, self.wuk, compute_kernel_config=self.ckc, memory_config=mem)  # [1,Hd,Tp,512]
        ttnn.deallocate(q_nope)
        q_rope = self.rope(q_rope, cos, sin, mem)
        # ---- kv latent row of these tokens -> the cache
        kv = ttnn.matmul(x, self.wkva, compute_kernel_config=self.ckc, memory_config=mem)  # [1,1,Tp,576]
        c = self.kv_norm(ttnn.slice(kv, (0, 0, 0, 0), (1, 1, Tp, self.kvl), memory_config=mem), mem)
        kr = self.rope(ttnn.slice(kv, (0, 0, 0, self.kvl), (1, 1, Tp, cfg.latent_width), memory_config=mem), cos, sin, mem)
        ttnn.deallocate(kv)
        zeros = ttnn.zeros((1, 1, Tp, SINK_COLS), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                           memory_config=mem)
        row = ttnn.concat([c, kr, zeros], dim=-1, memory_config=mem)  # [1,1,Tp,608] TILE
        ttnn.deallocate(c)
        ttnn.deallocate(kr)
        row_rm = ttnn.to_layout(row, ttnn.ROW_MAJOR_LAYOUT, memory_config=mem)
        ttnn.deallocate(row)
        self._write_rows(row_rm, p0, T)
        ttnn.deallocate(row_rm)
        # ---- indexer (sparse layers past the dense regime)
        selected = None
        if self.indexer is not None:
            selected = self.indexer.forward(x, q_resid, p0, T, mem)
        ttnn.deallocate(q_resid)
        # ---- sparse attention over the index lists
        qcat = ttnn.concat([q_lat, q_rope, ttnn.zeros((1, self.Hd, Tp, SINK_COLS), dtype=ttnn.bfloat16,
                                                      layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=mem)],
                           dim=-1, memory_config=mem)
        ttnn.deallocate(q_lat)
        ttnn.deallocate(q_rope)
        q_rm = ttnn.to_layout(qcat, ttnn.ROW_MAJOR_LAYOUT, memory_config=mem)
        ttnn.deallocate(qcat)
        q_rm = self._add_sink_cols(q_rm, Tp, mem)
        if self.Hp != self.Hd:
            pad = ttnn.zeros((1, self.Hp - self.Hd, Tp, self.width), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                             device=mesh, memory_config=mem)
            q_rm = ttnn.concat([q_rm, pad], dim=1, memory_config=mem)
        idx = upload(mesh, self.index_lists(p0, T, selected), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        o = ttnn.transformer.sparse_sdpa(q_rm, self.cache, idx, self.kvl, kv_format=SKF.BF16, scale=self.scale,
                                         k_chunk_size=K_CHUNK, compute_kernel_config=self.ckc)  # [1,Hp,Tp,512] RM
        ttnn.deallocate(q_rm)
        ttnn.deallocate(idx)
        o = ttnn.slice(o, (0, 0, 0, 0), (1, self.Hd, Tp, self.kvl), memory_config=mem)
        o = ttnn.to_layout(o, ttnn.TILE_LAYOUT, memory_config=mem)
        ov = ttnn.matmul(o, self.wuvT, compute_kernel_config=self.ckc, memory_config=mem)  # [1,Hd,Tp,256]
        ttnn.deallocate(o)
        ov = ttnn.reshape(ttnn.permute(ov, (0, 2, 1, 3), memory_config=mem), (1, 1, Tp, self.Hd * self.vd))
        g = ttnn.matmul(x, self.wg, compute_kernel_config=self.ckc, memory_config=mem)
        gated = ttnn.multiply(ov, ttnn.sigmoid(g, memory_config=mem), memory_config=mem)
        ttnn.deallocate(ov)
        ttnn.deallocate(g)
        part = ttnn.matmul(gated, self.wo, compute_kernel_config=self.ckc, memory_config=mem)
        ttnn.deallocate(gated)
        full = self.ar(part, mem)
        ttnn.deallocate(part)
        return self.post_norm(full, mem)

    def _add_sink_cols(self, q_rm, Tp, mem):
        """q's sink columns (cols 576..607, zero so far) += this die's [s_hi, s_lo] per head (broadcast over rows)."""
        lead = ttnn.slice(q_rm, (0, 0, 0, 0), (1, self.Hd, Tp, self.cfg.latent_width), memory_config=mem)
        qs = ttnn.slice(self.q_sink, (0, 0, 0, 0), (1, self.Hd, 1, SINK_COLS))
        qs = ttnn.repeat(qs, ttnn.Shape((1, 1, Tp, 1)), memory_config=mem)
        out = ttnn.concat([lead, qs], dim=-1, memory_config=mem)
        ttnn.deallocate(q_rm)
        return out

    def _write_rows(self, row_rm, p0: int, T: int):
        """Write rows [0, T) of row_rm [1,1,Tp,608] RM to the cache at positions p0.. (ring rows for SWA)."""
        start = p0 if self.kind == "sparse" else p0 % self.rows
        first = min(T, self.rows - start)
        segs = [(0, start, first)]
        if first < T:  # SWA ring wrap
            segs.append((first, 0, T - first))
        for src, dst, n in segs:
            part = row_rm if (src == 0 and n == int(row_rm.shape[-2])) else ttnn.slice(
                row_rm, (0, 0, src, 0), (1, 1, src + n, self.width))
            ttnn.experimental.slice_write(part, self.cache, [0, 0, dst, 0], [1, 1, dst + n, self.width], [1, 1, 1, 1])
            if part is not row_rm:
                ttnn.deallocate(part)
