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
        self.q_sink = upload(mesh, qs[:, :Hd].reshape(N, Hd, 1, SINK_COLS), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                             mapper=("shard", 0))
        self.stage = None  # the decode staging window (init_stage)
        # caches (row-major bf16, replicated): sparse layers hold every position + a sink block at the end; SWA layers
        # a ring of `ring` rows + the sink block
        if self.kind == "sparse":
            self.rows = -(-max_ctx // 32) * 32 + 64  # + the decode staging window's lookahead (two 32-row slabs)
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

    # ------------------------------------------------------------------ shared pieces
    _ZEROS = {}

    def _zeros(self, shape):
        """A persistent zero tensor of this shape (shared by the layers; created outside any trace capture: the decode
        path's first pass of each shape is eager)."""
        key = (id(self.mesh), tuple(shape))
        z = Hy4Attention._ZEROS.get(key)
        if z is None:
            z = Hy4Attention._ZEROS[key] = ttnn.zeros(tuple(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                                       device=self.mesh)
        return z
    def _q(self, x, cos, sin, mem):
        """-> (q_lat [1,Hd,Tp,512], q_rope [1,Hd,Tp,64] roped, q_resid [1,1,Tp,1024])."""
        Tp = int(x.shape[-2])
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
        return q_lat, q_rope, q_resid

    def _kv_row(self, x, cos, sin, mem):
        """-> the cache rows of these tokens [1,1,Tp,608] TILE: [norm(c) | rope(k_rope) | 0]."""
        Tp = int(x.shape[-2])
        kv = ttnn.matmul(x, self.wkva, compute_kernel_config=self.ckc, memory_config=mem)  # [1,1,Tp,576]
        c = self.kv_norm(ttnn.slice(kv, (0, 0, 0, 0), (1, 1, Tp, self.kvl), memory_config=mem), mem)
        kr = self.rope(ttnn.slice(kv, (0, 0, 0, self.kvl), (1, 1, Tp, self.cfg.latent_width), memory_config=mem),
                       cos, sin, mem)
        ttnn.deallocate(kv)
        row = ttnn.concat([c, kr, self._zeros((1, 1, Tp, SINK_COLS))], dim=-1, memory_config=mem)
        ttnn.deallocate(c)
        ttnn.deallocate(kr)
        return row

    def _attend(self, x, q_lat, q_rope, idx, mem):
        """sparse_sdpa over idx [1,1,Tp,K] uint32, then V-up, output gate, o_proj, all-reduce, post norm."""
        Tp = int(x.shape[-2])
        qs = ttnn.repeat(self.q_sink, ttnn.Shape((1, 1, Tp, 1)), memory_config=mem)  # [1,Hd,Tp,32]
        qcat = ttnn.concat([q_lat, q_rope, qs], dim=-1, memory_config=mem)  # [1,Hd,Tp,608]
        ttnn.deallocate(qs)
        if self.Hp != self.Hd:
            q2 = ttnn.concat([qcat, self._zeros((1, self.Hp - self.Hd, Tp, self.width))], dim=1, memory_config=mem)
            ttnn.deallocate(qcat)
            qcat = q2
        q_rm = ttnn.to_layout(qcat, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # sparse_sdpa: DRAM
        ttnn.deallocate(qcat)
        o = ttnn.transformer.sparse_sdpa(q_rm, self.cache, idx, self.kvl, kv_format=SKF.BF16, scale=self.scale,
                                         k_chunk_size=K_CHUNK, compute_kernel_config=self.ckc)  # [1,Hp,Tp,512] RM
        ttnn.deallocate(q_rm)
        o2 = ttnn.slice(o, (0, 0, 0, 0), (1, self.Hd, Tp, self.kvl), memory_config=mem)
        ttnn.deallocate(o)
        o = ttnn.to_layout(o2, ttnn.TILE_LAYOUT, memory_config=mem)
        ttnn.deallocate(o2)
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

    # ------------------------------------------------------------------ eager (prefill)
    def forward(self, x, p0: int, T: int, cos_sin, mem=None):
        """x: [1, 1, Tp, 3072] bf16 replicated (normed attention input), rows p0 .. p0 + T - 1 real (Tp = T padded to
        32). cos_sin: {kind: (cos, sin)} fp32 [1, 1, Tp, 64] device tables of these positions (shared by the layers).
        Returns [1, 1, Tp, 3072] bf16 replicated."""
        mem = mem or ttnn.DRAM_MEMORY_CONFIG
        cos, sin = cos_sin[self.kind]
        q_lat, q_rope, q_resid = self._q(x, cos, sin, mem)
        row = self._kv_row(x, cos, sin, mem)
        row_rm = ttnn.to_layout(row, ttnn.ROW_MAJOR_LAYOUT, memory_config=mem)
        ttnn.deallocate(row)
        self._write_rows(row_rm, p0, T)
        ttnn.deallocate(row_rm)
        selected = self.indexer.forward(x, q_resid, p0, T, mem) if self.indexer is not None else None
        ttnn.deallocate(q_resid)
        idx = upload(self.mesh, self.index_lists(p0, T, selected), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        out = self._attend(x, q_lat, q_rope, idx, mem)
        for t_ in (q_lat, q_rope, idx):
            ttnn.deallocate(t_)
        return out

    # ------------------------------------------------------------------ trace-safe step (decode / verify rows)
    def init_stage(self, base: int):
        """After a prefill: the 64-row staging window starting at position `base` (32-aligned) from the cache."""
        r0 = base if self.kind == "sparse" else base % self.rows
        if r0 + 64 <= self.rows:
            w = ttnn.slice(self.cache, (0, 0, r0, 0), (1, 1, r0 + 64, self.width))
        else:  # the SWA ring wraps inside the window
            a = ttnn.slice(self.cache, (0, 0, r0, 0), (1, 1, self.rows, self.width))
            b = ttnn.slice(self.cache, (0, 0, 0, 0), (1, 1, r0 + 64 - self.rows, self.width))
            w = ttnn.concat([a, b], dim=2)
            ttnn.deallocate(a)
            ttnn.deallocate(b)
        wt = ttnn.to_layout(w, ttnn.TILE_LAYOUT)
        ttnn.deallocate(w)
        if self.stage is None:
            self.stage = wt
        else:
            ttnn.copy(wt, self.stage)
            ttnn.deallocate(wt)

    def step(self, x, si, mem=None):
        """Trace-safe rows pass. si: the step inputs (tt/decode.StepInputs): cos/sin, staging maps, slab offsets,
        index lists. The KV rows go through the staging window and two 32-row slab writes."""
        mem = mem or ttnn.DRAM_MEMORY_CONFIG
        cos, sin = si.cos_sin[self.kind]
        q_lat, q_rope, q_resid = self._q(x, cos, sin, mem)
        row = self._kv_row(x, cos, sin, mem)
        a = ttnn.matmul(si.A, self.stage, compute_kernel_config=self.ckc, memory_config=mem)
        b = ttnn.matmul(si.P, row, compute_kernel_config=self.ckc, memory_config=mem)
        ttnn.deallocate(row)
        new = ttnn.add(a, b, memory_config=mem)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        ttnn.copy(new, self.stage)
        ttnn.deallocate(new)
        st_rm = ttnn.to_layout(self.stage, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        offs = si.off_sparse if self.kind == "sparse" else si.off_swa
        for j in range(2):
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                self.cache, st_rm, si.slot, offs[j], 0, 1, 0, input_row_offset=32 * j, input_rows=32)
        ttnn.deallocate(st_rm)
        if self.kind == "swa":
            idx = si.idx_swa
        elif self.indexer is not None:
            idx = self.indexer.step(x, q_resid, si, mem)  # the selection, or si.idx_dense in the dense regime
        else:
            idx = si.idx_dense
        ttnn.deallocate(q_resid)
        out = self._attend(x, q_lat, q_rope, idx, mem)
        ttnn.deallocate(q_lat)
        ttnn.deallocate(q_rope)
        if idx is not si.idx_swa and idx is not si.idx_dense:
            ttnn.deallocate(idx)
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
