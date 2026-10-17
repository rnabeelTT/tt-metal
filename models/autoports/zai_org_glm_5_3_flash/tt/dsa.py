# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash DeepSeek sparse attention (DSA) layer: NoPE MLA over a 512-wide latent + lightning indexer.

HF (`Glm5NextTextAttention` + `Glm5NextTextIndexer`), 64 heads, no positional encoding anywhere:
    q_r  = rmsnorm(q_a(x))                          [T, 1536] (shared with the indexer)
    q    = q_b(q_r)  -> [T, 64, 256]
    c    = rmsnorm(kv_a(x))                         [T, 512]  (the cached latent)
    k_h  = W_UK_h c,  v_h = W_UV_h c                (kv_b rows h*512 + [0,256) / [256,512))
    o_h  = softmax(q_h k_h^T / 16 + M) v_h ;  out = o_proj(concat_h o_h)
M = the indexer selection. Per query position t: the complete 4-token pools p (pool p = tokens 4p..4p+3,
4p + 3 <= t), scored
    I[t,p] = sum_h w[t,h] * relu(qI[t,h] . poolK[p] / sqrt(128)),   w = weights_proj(x) / sqrt(32),
    poolK[p,c] = sum_j softmax_j(gate[4p+j,c] + ape[j,c]) * kI[4p+j,c],   kI = LayerNorm(wk x), gate = x @ Wg^T
the top-512 pools (2048 tokens) plus the raw tail of t's incomplete pool. A query with <= 512 complete pools
(t <= 2050) attends to every visible key (exact dense causal attention).

Absorbed form (identical math): q_abs_h = W_UK_h^T q_h [512]; MQA attention of q_abs over latent rows (K = V =
latent, scale 1/16); o_h = W_UV_h o_lat.

Paged device state per layer (S decode slots, max context T per slot, NB page blocks of 128 tokens):
    page_table [S, T/128 + 1] int32 ROW_MAJOR   (block of slot s's tokens [128 j, 128 j + 128); the extra column is the
                                                 indexer's one-tile overrun and points to a reserved dummy block)
    latent   [1, 1, NB*128, 512] bf16 ROW_MAJOR  (block b = rows [128 b, 128 b + 128); written with ttnn.scatter_rows)
    pooled   [NB', 1, 32, 128] bf16 TILE         (block b = the 32 pool keys of the same 128 tokens; NB' >= NB)
    pool_acc 3 x [1, 1, S, 128] fp32             (online-softmax state m / den / num of each slot's open pool)
Every index list is LOGICAL (slot-local token / pool positions); the ops map them through the page table
(sparse_sdpa page_table, indexer_score_dsa page_table, dsa_decode_scores, paged_fill_cache / paged_update_cache).
Attention for every query row is ttnn.transformer.sparse_sdpa with an explicit index list: [0..t] for dense rows,
the expanded top pools + tail for sparse rows (sentinel 0xFFFFFFFF tail), except the first prefill chunk of a
dense prompt, which takes flash_mla_prefill on its own latent. Decode slots at position -1 are inactive: their caches
and open-pool state are not changed.
"""

from __future__ import annotations

import os

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

from .weight_cache import cached

DRAM = ttnn.DRAM_MEMORY_CONFIG
DENSE_MAX_POSITION = 2050  # query positions <= this attend to every visible key
BLOCK = 128  # page block (tokens); its 32 pool keys are one tile row of the pooled-key cache
KPOOL = 4
TOPK_POOLS = 512
IDX_WIDTH = 2080  # >= 2048 + 3 tail entries, multiple of the sparse_sdpa k chunk
SENTINEL_F = 4294967296.0  # fp32 -> uint32 typecast saturates this to 0xFFFFFFFF
NEG = -1.0e30


class GlmDSA(LightweightModule):
    def __init__(
        self,
        mesh_device,
        tc: dict,
        sd: dict[str, torch.Tensor],
        layer_idx: int,
        *,
        max_batch: int = 32,
        max_context: int = 4096,
        weight_dtype=ttnn.bfloat8_b,
        num_blocks: int | None = None,  # page blocks (incl. the dummy block); default: every slot at max_context
        block_size: int = BLOCK,
        weight_cache=None,  # the layer's tt/weight_cache.WeightCache (sd is read only on a miss)
        bmm_dtype=None,  # W_UK / W_UV dtype of the precision policy (retyped on device at load; None = bf16)
    ):
        self.mesh = mesh_device
        self.layer_idx = layer_idx
        self.tp = tuple(mesh_device.shape)[1]
        # TP (multichip decoder): each device owns H / tp query heads (q_b columns, W_UK / W_UV heads, o_proj rows); the
        # latent cache, the pooled index keys and the indexer are replicated (computed on every device), so top-k pool
        # selection needs no collective. With tp > 1 the output projection returns per-device partial sums (the caller
        # reduces).
        H_all = tc["num_attention_heads"]
        assert H_all % self.tp == 0, (H_all, self.tp)
        self.H = H_all // self.tp  # local query heads
        self.Dn = tc["qk_nope_head_dim"]  # 256
        self.Dv = tc["v_head_dim"]  # 256
        self.R = tc["kv_lora_rank"]  # 512
        self.hidden = tc["hidden_size"]
        self.eps = tc["rms_norm_eps"]
        self.scale = self.Dn**-0.5  # 1/16
        self.IH = tc["index_n_heads"]  # 32
        self.ID = tc["index_head_dim"]  # 128
        assert tc["index_kpool"] == KPOOL and tc["index_topk"] == TOPK_POOLS * KPOOL
        self.S = max_batch
        assert max_context % 128 == 0
        self.T = max_context
        assert block_size == BLOCK and max_context % BLOCK == 0, "page blocks are 128 tokens (32 pools = one tile row)"
        assert max_context < 1 << 24, "logical positions are carried in fp32"
        self.P = max_context // KPOOL  # pool slots per user
        self.P_alloc = self.P + 32  # + one tile: indexer_score_dsa causal start sits at P (all real pools visible)
        self.MB = self.P_alloc // 32  # page-table width: T/128 blocks + the indexer's overrun block
        self.NB = num_blocks if num_blocks is not None else self.S * (max_context // BLOCK) + 1
        self.dummy_block = self.NB - 1  # never handed to a slot; target of the overrun column
        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        self._rep = rep

        def w(name, t, dtype=weight_dtype, shard_dim=None, retype=None):
            mapper = rep if shard_dim is None or self.tp == 1 else ttnn.ShardTensorToMesh(mesh_device, dim=shard_dim)

            def build():
                out = ttnn.from_torch(
                    t().contiguous(),
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=DRAM,
                    mesh_mapper=mapper,
                )
                if retype is not None and retype != dtype:
                    out2 = ttnn.typecast(out, retype)
                    ttnn.deallocate(out)
                    out = out2
                return out

            return cached(weight_cache, f"dsa.{name}", build)

        def norm_w(t):
            return ttnn.from_torch(
                t.to(torch.bfloat16).reshape(1, 1, -1, 32),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                mesh_mapper=rep,
            )

        a = "self_attn."
        self.w_q_a = w("q_a", lambda: sd[a + "q_a_proj.weight"].T)
        self.q_a_norm = norm_w(sd[a + "q_a_layernorm.weight"])
        self.w_q_b = w("q_b", lambda: sd[a + "q_b_proj.weight"].T, shard_dim=-1)  # [1536, 64*256], head-major columns
        self.w_kv_a = w("kv_a", lambda: sd[a + "kv_a_proj_with_mqa.weight"].T)  # [4096, 512]
        self.kv_a_norm = norm_w(sd[a + "kv_a_layernorm.weight"])
        kvb = lambda: sd[a + "kv_b_proj.weight"].float().reshape(H_all, self.Dn + self.Dv, self.R)
        # q_abs = q @ W_UK; o = o_lat @ W_UV^T (per head)
        self.w_uk = w(
            "w_uk",
            lambda: kvb()[:, : self.Dn, :].reshape(1, H_all, self.Dn, self.R),
            ttnn.bfloat16,
            shard_dim=1,
            retype=bmm_dtype,
        )
        self.w_uv_t = w(
            "w_uv_t",
            lambda: kvb()[:, self.Dn :, :].transpose(1, 2).reshape(1, H_all, self.R, self.Dv),
            ttnn.bfloat16,
            shard_dim=1,
            retype=bmm_dtype,
        )
        self.w_o = w("o", lambda: sd[a + "o_proj.weight"].T, shard_dim=-2)  # [64*256, 4096], head-major rows
        self.bmm_reuse = False  # per-head matmuls on an explicit full-grid reuse config (optimized decoder)
        self.decode_mm = None  # (x, w, role, ckc, mc) -> out | None: decode program-config hook (optimized decoder)
        # head split / concat with the nlp_create_qkv_heads / nlp_concat_heads data-movement ops instead of reshape +
        # permute (fused decoder; bit-identical, tools/fusing_probe.py)
        self.heads_op = False
        self.decode_rm_relayout = False  # decode head relayouts through row-major (fused decoder; _rm_rows)
        # indexer (bf16 in the checkpoint; kept bf16)
        ix = a + "indexer."
        self.w_iq = w("iq", lambda: sd[ix + "wq_b.weight"].T, ttnn.bfloat16)  # [1536, 32*128]
        self.w_ik = w("ik", lambda: sd[ix + "wk.weight"].T, ttnn.bfloat16)  # [4096, 128]
        self.ik_norm_w = w("ik_norm_w", lambda: sd[ix + "k_norm.weight"].reshape(1, 1, 1, -1), ttnn.bfloat16)
        self.ik_norm_b = w("ik_norm_b", lambda: sd[ix + "k_norm.bias"].reshape(1, 1, 1, -1), ttnn.bfloat16)
        # head weights with both scales folded: relu(a x) = a relu(x) for a = 1/sqrt(128) > 0
        self.w_iw = w(
            "iw",
            lambda: sd[ix + "weights_proj.weight"].float().T * (self.IH**-0.5) * (self.ID**-0.5),
            ttnn.bfloat16,
        )
        self.w_ig = w("ig", lambda: sd[ix + "index_kpool_compress_gate"].T, ttnn.bfloat16)  # [4096, 128]
        self.ape = sd[ix + "index_kpool_compress_ape"].float()  # [4, 128] host copy
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.sdpa_ckc = self.ckc
        self._const_cache = {}
        self.latent = None

    # ----------------------------------------------------------------------------------------------- caches
    def allocate_decode_state(self, batch: int | None = None, num_blocks: int | None = None):
        """num_blocks (vLLM serving): page blocks incl. the dummy block, known only after construction; overrides
        the constructor's count when given before the first allocation."""
        S = self.S
        if num_blocks is not None and self.latent is None:
            self.NB = int(num_blocks)
            self.dummy_block = self.NB - 1
        if self.latent is None:
            self.latent = ttnn.zeros(
                (1, 1, self.NB * BLOCK, self.R),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
                memory_config=DRAM,
            )
            # pool keys: >= NB blocks of 32 keys, sized as a whole number of P_alloc-long indexer views
            self.KB = -(-self.NB * 32 // self.P_alloc)  # indexer view batch
            self.pooled = ttnn.zeros(
                (self.KB * self.P_alloc // 32, 1, 32, self.ID),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=DRAM,
            )
            # a pool smaller than S full contexts (vLLM) starts with every column on the dummy block
            fits = self.S * (self.T // BLOCK) + 1 <= self.NB
            pt0 = (
                self.default_page_table()
                if fits
                else self.complete_page_table(torch.zeros(self.S, 0, dtype=torch.int32))
            )
            self.page_table = self._dev(pt0, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            self.page_table_host = pt0
            z = lambda: ttnn.zeros(
                (1, 1, S, self.ID), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=DRAM
            )
            self.acc_m, self.acc_den, self.acc_num = z(), z(), z()
        return self.latent

    # ------------------------------------------------------------------------------------------- constants
    def default_page_table(self) -> torch.Tensor:
        """Slot s owns blocks [s * T/128, (s + 1) * T/128) (a contiguous allocation; any permutation is valid)."""
        bps = self.T // BLOCK
        assert self.S * bps + 1 <= self.NB, "num_blocks too small for the default allocation"
        pt = torch.arange(self.S * bps, dtype=torch.int32).reshape(self.S, bps)
        return self.complete_page_table(pt)

    def complete_page_table(self, pt: torch.Tensor) -> torch.Tensor:
        """[S, n <= T/128] block ids -> [S, MB]: unused and overrun columns point to the dummy block."""
        out = torch.full((self.S, self.MB), self.dummy_block, dtype=torch.int32)
        out[: pt.shape[0], : pt.shape[1]] = pt.to(torch.int32)
        assert int(out.max()) < self.NB and int(out.min()) >= 0, "page-table block ids out of range"
        return out

    def set_page_table(self, pt: torch.Tensor):
        """Install a block table [S, <= T/128] (host -> the fixed device buffer, outside traces). Every later prefill /
        decode input is built against it; the blocks of a slot must not be changed while the slot holds a sequence."""
        full = self.complete_page_table(pt)
        self.page_table_host = full
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(full, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self._rep),
            self.page_table,
        )

    def _phys_rows(self, slot_of_row: torch.Tensor, pos: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Physical latent rows of logical (slot, position) pairs. Every position written / attended must sit in a block
        the slot owns (never the shared dummy block, which only absorbs the indexer's overrun reads)."""
        blk = self.page_table_host[slot_of_row, pos // BLOCK].long()
        assert not bool(
            (valid & (blk == self.dummy_block)).any()
        ), "a cache position maps to the dummy block (page table)"
        return blk * BLOCK + pos % BLOCK

    def _const(self, key, fn=None):
        t = self._const_cache.get(key)
        if t is None:
            assert fn is not None, f"DSA constant {key} was not created by prepare_constants()"
            t = fn()
            self._const_cache[key] = t
        return t

    def _dev(self, t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=self.mesh, mesh_mapper=self._rep)

    # ------------------------------------------------------------------------------------------ projections
    def _rms(self, x, w):
        return ttnn.rms_norm(x, epsilon=self.eps, weight=w, memory_config=DRAM, compute_kernel_config=self.ckc)

    def _mm(self, x, w, **kw):
        return ttnn.linear(x, w, compute_kernel_config=self.ckc, memory_config=DRAM, **kw)

    def _bmm(self, x, w, rows: int):
        """Per-head matmul [1,H,rows,K] x [1,H,K,N]. With bmm_reuse (decode): one head per core, the full N per core (the
        default config puts 64 heads on 8 - 16 cores; tools/dsa_bmm_probe.py)."""
        kw = {}
        # decode rows only (one tile of users): the configuration was measured there; a 2048-row prefill on it hung the
        # device (2026-10-16), and prefill is not weight-read bound
        if self.bmm_reuse and rows <= 32:
            grid = self.mesh.compute_with_storage_grid_size()
            Kt, Nt, Mt = w.shape[-2] // 32, w.shape[-1] // 32, -(-rows // 32)
            kw["program_config"] = ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=(grid.x, grid.y),
                in0_block_w=min(Kt, 8),
                out_subblock_h=1,
                out_subblock_w=max(d for d in (4, 2, 1) if Nt % d == 0),
                per_core_M=Mt,
                per_core_N=Nt,
            )
        return ttnn.matmul(x, w, compute_kernel_config=self.ckc, memory_config=DRAM, **kw)

    def _heads_op_rows(self, rows: int) -> bool:
        """nlp_create_qkv_heads / nlp_concat_heads split their work by 32-row blocks: one decode tile of rows runs on one
        core and is slower than reshape + permute (H 64: 209 vs 187 us), so only prefill rows take them."""
        return self.heads_op and rows % 32 == 0 and rows > 32

    def _rm_rows(self, rows: int) -> bool:
        """Decode tile of users (32 rows): the head relayout goes through row-major (untilize, row-major permute, tilize)
        instead of the tiled reshape + permute (H 64: split 187 -> 52 us, concat 186 -> 63 us; H 16: 50 -> 41 / 39 us;
        bit-identical, tools/fusing_probe.py relayout). One row stays tiled (25 vs 40 us)."""
        return self.decode_rm_relayout and rows == 32

    def _split_heads(self, t, rows: int, heads: int, dim: int):
        """[1,1,rows,heads*dim] (head-major columns) -> [1,heads,rows,dim]."""
        if self._heads_op_rows(rows):
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                t, num_heads=heads, num_kv_heads=0, transpose_k_heads=False, memory_config=DRAM
            )
            for e in (k, v):
                ttnn.deallocate(e)
            return q
        if self._rm_rows(rows):
            rm = ttnn.reshape(ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM), (1, rows, heads, dim))
            p = ttnn.permute(rm, (0, 2, 1, 3), memory_config=DRAM)
            ttnn.deallocate(rm)
            out = ttnn.to_layout(p, ttnn.TILE_LAYOUT, memory_config=DRAM)
            ttnn.deallocate(p)
            return out
        return ttnn.permute(ttnn.reshape(t, (1, rows, heads, dim)), (0, 2, 1, 3), memory_config=DRAM)

    def _concat_heads(self, t, rows: int):
        """[1,H,rows,Dv] -> [1,1,rows,H*Dv] (head-major columns)."""
        if self._heads_op_rows(rows):
            # nlp_concat_heads holds every head of a 32-row block in L1 (double-buffered): 64 x 256 needs 2.2 MB > 1.5 MB,
            # so groups of <= 16 heads (a free leading-dim view; 4 groups beat 2: 1043 vs 1101 us at 2048 rows), then the
            # group outputs side by side
            groups = max(1, self.H // 16)
            if groups == 1:
                return ttnn.experimental.nlp_concat_heads(t, memory_config=DRAM)
            hg = self.H // groups
            x = ttnn.experimental.nlp_concat_heads(ttnn.reshape(t, (groups, hg, rows, self.Dv)), memory_config=DRAM)
            parts = [
                ttnn.slice(x, (i, 0, 0, 0), (i + 1, 1, rows, hg * self.Dv), memory_config=DRAM) for i in range(groups)
            ]
            ttnn.deallocate(x)
            out = ttnn.concat(parts, dim=-1, memory_config=DRAM)
            for p_ in parts:
                ttnn.deallocate(p_)
            return out
        if self._rm_rows(rows):
            rm = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
            p = ttnn.permute(rm, (0, 2, 1, 3), memory_config=DRAM)
            ttnn.deallocate(rm)
            out = ttnn.to_layout(ttnn.reshape(p, (1, 1, rows, self.H * self.Dv)), ttnn.TILE_LAYOUT, memory_config=DRAM)
            ttnn.deallocate(p)
            return out
        return ttnn.reshape(ttnn.permute(t, (0, 2, 1, 3), memory_config=DRAM), (1, 1, rows, self.H * self.Dv))

    def _q_abs(self, x, rows: int):
        """x [1,1,rows,hidden] -> (q_abs [1, H, rows, 512], q_r [1,1,rows,1536])."""
        qr = self._rms(self._mm(x, self.w_q_a), self.q_a_norm)
        q = self._mm(qr, self.w_q_b)  # [1,1,rows,H*256]
        q = self._split_heads(q, rows, self.H, self.Dn)
        q_abs = self._bmm(q, self.w_uk, rows)  # [1,H,rows,512]
        ttnn.deallocate(q)
        return q_abs, qr

    def _latent(self, x):
        return self._rms(self._mm(x, self.w_kv_a), self.kv_a_norm)

    def _index_k(self, x):
        """kI = LayerNorm(wk x) [1,1,rows,128] (bf16), gate logits gl = x Wg^T [1,1,rows,128] (fp32)."""
        k = ttnn.layer_norm(
            self._mm(x, self.w_ik),
            epsilon=1e-6,
            weight=self.ik_norm_w,
            bias=self.ik_norm_b,
            compute_kernel_config=self.ckc,
        )
        gl = self._mm(x, self.w_ig, dtype=ttnn.float32)
        return k, gl

    def _out(self, o_lat, rows: int):
        """o_lat [1, H, rows, 512] (TILE) -> [1,1,rows,hidden]."""
        o = self._bmm(o_lat, self.w_uv_t, rows)  # [1,H,rows,256]
        o = self._concat_heads(o, rows)
        out = self.decode_mm(o, self.w_o, "o", self.ckc, DRAM) if self.decode_mm is not None else None
        return out if out is not None else self._mm(o, self.w_o)

    # ---------------------------------------------------------------------------------------- pooled keys
    def _pool_chunk(self, k, gl, n: int, vmask):
        """Pools of one prefill chunk (n rows, pools aligned to the chunk start, rows >= valid are padding).
        Returns (poolK [1,1,n/4,128] bf16, den, num, m [1,1,n/4,128] fp32): poolK = num / den, each pool's softmax
        shifted by its own per-channel maximum m (as HF: every exp argument <= 0 and the pool's largest term is 1,
        so no pool underflows whatever the spread of the gate logits across the chunk)."""
        Sm = self._const(("poolsum", n))
        ape_rows = self._const(("ape_rows", n))
        L = ttnn.add(gl, ape_rows, memory_config=DRAM)
        Lm = ttnn.add(ttnn.multiply(L, vmask), ttnn.multiply(ttnn.rsub(vmask, 1.0), NEG))  # padding rows -> -1e30
        # per-pool max: regroup the 4 rows of each pool side by side ([n, 128] -> [n/4, 512]), max over the 4 slices
        g = ttnn.to_layout(
            ttnn.reshape(ttnn.to_layout(Lm, ttnn.ROW_MAJOR_LAYOUT), (1, 1, n // KPOOL, KPOOL * self.ID)),
            ttnn.TILE_LAYOUT,
        )
        m = ttnn.slice(g, (0, 0, 0, 0), (1, 1, n // KPOOL, self.ID))
        for j in range(1, KPOOL):
            part = ttnn.slice(g, (0, 0, 0, j * self.ID), (1, 1, n // KPOOL, (j + 1) * self.ID))
            m2 = ttnn.maximum(m, part, memory_config=DRAM)
            ttnn.deallocate(m)
            ttnn.deallocate(part)
            m = m2
        ttnn.deallocate(g)
        m_rows = ttnn.repeat_interleave(m, KPOOL, dim=2)  # exact [n, 128]
        e = ttnn.exp(ttnn.subtract(Lm, m_rows, memory_config=DRAM), memory_config=DRAM)
        ek = ttnn.multiply(e, ttnn.typecast(k, ttnn.float32), memory_config=DRAM)
        den = ttnn.matmul(Sm, e, compute_kernel_config=self.ckc, memory_config=DRAM, dtype=ttnn.float32)
        num = ttnn.matmul(Sm, ek, compute_kernel_config=self.ckc, memory_config=DRAM, dtype=ttnn.float32)
        poolK = ttnn.typecast(ttnn.divide(num, den), ttnn.bfloat16)  # den >= 1 (the pool max term)
        for t in (L, Lm, m_rows, e, ek):
            ttnn.deallocate(t)
        return poolK, den, num, m

    # --------------------------------------------------------------------------------------- index lists
    def _sparse_rows(self, top_pools, pos_dev, rows: int):
        """top_pools TILE [1,1,rows,512] pool ids; pos_dev fp32 [1,1,rows,1] query positions -> logical token indices
        fp32 [1,1,rows,IDX_WIDTH] (sentinels at the end)."""
        n_tok = TOPK_POOLS * KPOOL
        offs = self._const("pool_offs")
        p = ttnn.typecast(top_pools, ttnn.float32)
        # tokens of every selected pool: 4p + j. Elementwise (exact in fp32); a [512 -> 2048] expansion matmul is not:
        # the matmul unit reads fp32 operands at TF32 precision (pool ids >= 2**11 off by up to 508 tokens at HiFi4)
        p4 = ttnn.repeat_interleave(p, KPOOL, dim=3)
        ttnn.deallocate(p)
        tok = ttnn.add(ttnn.multiply(p4, float(KPOOL)), offs)
        ttnn.deallocate(p4)
        # tail: tokens 4*floor((t+1)/4) .. t (0-3 of them), then sentinels up to IDX_WIDTH
        tail_cols = IDX_WIDTH - n_tok
        j = self._const(("tail_j", tail_cols))
        start = ttnn.multiply(ttnn.floor(ttnn.multiply(ttnn.add(pos_dev, 1.0), 1.0 / KPOOL)), float(KPOOL))
        tail_tok = ttnn.add(start, j)  # [1,1,rows,tail_cols]
        tail_ok = ttnn.le(tail_tok, pos_dev)
        tail = ttnn.where(tail_ok, tail_tok, SENTINEL_F)
        out = ttnn.concat([tok, tail], dim=-1)
        for t in (tok, tail, tail_tok, tail_ok, start):
            ttnn.deallocate(t)
        return out

    def _select_pools(self, scores, limit_dev):
        """scores [1,1,rows,P'] ; limit_dev fp32 [1,1,rows,1] = #complete visible pools of each row
        -> top-512 pool ids (TILE) [1,1,rows,512]."""
        Pw = scores.shape[-1]
        iota = self._const(("iota", Pw))
        if scores.layout != ttnn.TILE_LAYOUT:
            scores = ttnn.to_layout(scores, ttnn.TILE_LAYOUT)
        if scores.dtype != ttnn.bfloat16:
            scores = ttnn.typecast(scores, ttnn.bfloat16)
        ok = ttnn.typecast(ttnn.lt(iota, limit_dev), ttnn.bfloat16)
        masked = ttnn.where(ok, scores, float("-inf"))
        ttnn.deallocate(ok)
        _, ids = ttnn.topk(masked, TOPK_POOLS, dim=-1, largest=True, sorted=True)
        ttnn.deallocate(masked)
        return ids

    def _sparse_attend(self, q_abs, index_f32, cache_batch_idx=None):
        """q_abs [1,H,rows,512] TILE; index_f32 [1,1,rows,IDX_WIDTH] fp32 logical positions (sentinel 2**32) ->
        o_lat TILE [1,H,rows,512]. Page-table row: cache_batch_idx (prefill) or the query row (decode, one per slot)."""
        idx = ttnn.to_layout(ttnn.typecast(index_f32, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        H = q_abs.shape[1]
        pad = -H % 32  # sparse_sdpa takes whole 32-head tiles: TP leaves 16 heads per die -> zero heads, sliced off
        q_p = ttnn.pad(q_abs, [(0, 0), (0, pad), (0, 0), (0, 0)], 0.0) if pad else q_abs
        q_rm = ttnn.to_layout(q_p, ttnn.ROW_MAJOR_LAYOUT)
        if pad:
            ttnn.deallocate(q_p)
        o = ttnn.transformer.sparse_sdpa(
            q_rm,
            self.latent,
            idx,
            self.R,
            kv_format=ttnn.SparseKVFormat.BF16,
            scale=self.scale,
            k_chunk_size=32,
            compute_kernel_config=self.sdpa_ckc,
            cache_batch_idx=cache_batch_idx,
            page_table=self.page_table,
            page_block_size=BLOCK,
        )
        ttnn.deallocate(idx)
        ttnn.deallocate(q_rm)
        if pad:
            o_h = ttnn.slice(o, (0, 0, 0, 0), (o.shape[0], H, o.shape[2], o.shape[3]))
            ttnn.deallocate(o)
            o = o_h
        return ttnn.to_layout(o, ttnn.TILE_LAYOUT)

    def index_row_block(self, n: int) -> int:
        """Rows per indexer block of a prefill chunk: [rows, P_alloc] bf16 scores near GLM53_DSA_INDEX_BLOCK_MB (default 64
        MB; multiple of 32, divides n). The block's scores are live ~3x (scores, tiled copy, masked) and set the sparse
        prefill's DRAM peak (stage-6 memory: 128 MB blocks peaked 409 MiB per die at max_context 131072, the full
        model's largest prefill transient). Every row's selection is independent of the block: bit-identical."""
        target = int(float(os.environ.get("GLM53_DSA_INDEX_BLOCK_MB", "64")) * (1 << 20))
        rb = max(32, min(n, target // (2 * self.P_alloc) // 32 * 32))
        while n % rb:
            rb -= 32
        return rb

    def pooled_view(self):
        """The pool-key block pool as [KB, 1, P_alloc, 128]: the indexer's scored length is P_alloc; its key tile rows
        are read through the page table."""
        return ttnn.reshape(self.pooled, (self.KB, 1, self.P_alloc, self.ID))

    # ---------------------------------------------------------------------------------------------- prefill
    def prepare_constants(self, prefill_lengths):
        """Every constant a prefill / decode pass reads, created at setup (never inside a measured or traced pass)."""
        n_tok = TOPK_POOLS * KPOOL
        tail_cols = IDX_WIDTH - n_tok
        self._const(
            "pool_offs", lambda: self._dev(torch.arange(KPOOL).float().repeat(TOPK_POOLS).reshape(1, 1, 1, n_tok))
        )
        self._const(
            ("tail_j", tail_cols), lambda: self._dev(torch.arange(tail_cols).float().reshape(1, 1, 1, tail_cols))
        )
        self._const(
            ("iota", self.P_alloc), lambda: self._dev(torch.arange(self.P_alloc).float().reshape(1, 1, 1, self.P_alloc))
        )
        for n in prefill_lengths:
            npool = n // KPOOL
            self._const(
                ("poolsum", n),
                lambda: self._dev(torch.kron(torch.eye(npool), torch.ones(1, KPOOL)).reshape(1, 1, npool, n)),
            )
            self._const(("ape_rows", n), lambda: self._dev(self.ape.repeat(npool, 1).reshape(1, 1, n, self.ID)))

    def prefill_inputs(self, n: int, start: int, length: int, slot: int) -> dict:
        """Device inputs of one prefill chunk (host -> device input construction, outside the pass)."""
        assert start % 128 == 0 and n % 128 == 0 and start + n <= self.T, (start, n, self.T)
        ar = torch.arange(n)
        phys = self._phys_rows(torch.full((n,), slot), start + ar, ar < length)
        rows = torch.where(ar < length, phys, torch.full((n,), 0xFFFFFFFF)).reshape(1, n)
        pos = (start + ar).float().reshape(1, 1, n, 1)
        first_blk = start // KPOOL // 32
        nblk = n // KPOOL // 32
        inp = {
            "n": n,
            "start": start,
            "length": length,
            "slot": slot,
            "row_idx": self._dev(rows.to(torch.int64), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            "vmask": self._dev((ar < length).float().reshape(1, 1, n, 1)),
            "chunk_pt": self._dev(
                self.page_table_host[slot : slot + 1, first_blk : first_blk + nblk].contiguous(),
                ttnn.int32,
                ttnn.ROW_MAJOR_LAYOUT,
            ),
        }
        last = start + length - 1
        inp["path"] = (
            "flash"
            if (start == 0 and last <= DENSE_MAX_POSITION)
            else ("sparse" if last > DENSE_MAX_POSITION else "dense")
        )
        if inp["path"] != "flash":
            col = torch.arange(IDX_WIDTH).float().reshape(1, IDX_WIDTH)
            # logical positions of the slot; sparse_sdpa maps them through page-table row `slot`
            inp["dense_idx"] = self._dev(
                torch.where(col <= pos.reshape(n, 1), col, torch.full_like(col, SENTINEL_F)).reshape(1, 1, n, IDX_WIDTH)
            )
        if inp["path"] == "sparse":
            inp["pos"] = self._dev(pos)
            inp["limit"] = self._dev(torch.floor((pos + 1) / KPOOL))
            inp["is_dense"] = self._dev((pos <= DENSE_MAX_POSITION).float())
        return inp

    def prefill_forward(self, x, *, inputs):
        """One user's chunk x [1,1,n,hidden] at absolute positions [start, start+n) (start % 128 == 0; rows >= length
        are padding; inputs = prefill_inputs(...)); the user owns decode slot `slot`. Writes the latent rows, the
        chunk's pool keys and (for a chunk ending inside a pool) the slot's open-pool state.
        Returns (y [1,1,n,hidden], None)."""
        n = x.shape[-2]
        assert n == inputs["n"]
        length, slot = inputs["length"], inputs["slot"]
        q_abs, qr = self._q_abs(x, n)
        c = self._latent(x)  # [1,1,n,512] bf16 TILE
        c_rm = ttnn.to_layout(c, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.scatter_rows(self.latent, ttnn.reshape(c_rm, (n, self.R)), inputs["row_idx"])
        ttnn.deallocate(c_rm)
        # pool keys of this chunk: pools start at the chunk start (start % 4 == 0)
        k, gl = self._index_k(x)
        poolK, den, num, m = self._pool_chunk(k, gl, n, inputs["vmask"])
        ttnn.experimental.paged_fill_cache(self.pooled, poolK, page_table=inputs["chunk_pt"], batch_idx=0)
        if length % KPOOL:  # this chunk ends inside a pool: hand its partial state to the decode slot
            p_last = length // KPOOL
            d_row = ttnn.slice(den, (0, 0, p_last, 0), (1, 1, p_last + 1, self.ID))
            n_row = ttnn.slice(num, (0, 0, p_last, 0), (1, 1, p_last + 1, self.ID))
            m_row = ttnn.slice(m, (0, 0, p_last, 0), (1, 1, p_last + 1, self.ID))
            for buf, src in ((self.acc_den, d_row), (self.acc_num, n_row), (self.acc_m, m_row)):
                _write_row(buf, src, slot)
        for t in (poolK, den, num, m, k, gl):
            ttnn.deallocate(t)

        if inputs["path"] == "flash":
            o_lat = ttnn.transformer.flash_mla_prefill(
                q_abs,
                c,
                self.R,
                is_causal=True,
                scale=self.scale,
                compute_kernel_config=self.sdpa_ckc,
                memory_config=DRAM,
            )
        else:
            index = inputs["dense_idx"]
            owned = False
            if inputs["path"] == "sparse":
                # indexer scores over every pool slot of the user; rows that are still dense keep the dense list. Rows go
                # through the indexer in blocks that keep the [rows, P_alloc] score matrix near 128 MB (long contexts).
                qi_all = self._mm(qr, self.w_iq)  # [1,1,n,32*128]
                wi_all = self._mm(x, self.w_iw)  # [1,1,n,32]
                rb = self.index_row_block(n)
                parts = []
                for r0 in range(0, n, rb):
                    sl = lambda t, w: ttnn.slice(t, (0, 0, r0, 0), (1, 1, r0 + rb, w)) if rb < n else t
                    qi = sl(qi_all, self.IH * self.ID)
                    qi = self._split_heads(qi, rb, self.IH, self.ID)
                    wi = sl(wi_all, self.IH)
                    scores = ttnn.experimental.indexer_score_dsa(
                        qi,
                        self.pooled_view(),
                        wi,
                        chunk_start_idx=self.P,
                        cache_batch_idx=slot,
                        page_table=self.page_table,
                    )  # [1,1,rb,P_alloc] bf16 ROW_MAJOR
                    ids = self._select_pools(scores, sl(inputs["limit"], 1))
                    parts.append(self._sparse_rows(ids, sl(inputs["pos"], 1), rb))
                    for t in (qi, scores, ids):
                        ttnn.deallocate(t)
                    if rb < n:
                        ttnn.deallocate(wi)
                sparse = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=2)
                index = ttnn.where(inputs["is_dense"], inputs["dense_idx"], sparse)
                owned = True
                for t in [qi_all, wi_all, sparse] + (parts if len(parts) > 1 else []):
                    ttnn.deallocate(t)
            o_lat = self._sparse_attend(q_abs, index, cache_batch_idx=slot)
            if owned:
                ttnn.deallocate(index)
        ttnn.deallocate(q_abs)
        ttnn.deallocate(qr)
        ttnn.deallocate(c)
        y = self._out(o_lat, n)
        ttnn.deallocate(o_lat)
        return y, None

    # ----------------------------------------------------------------------------------------------- decode
    def _decode_host_inputs(self, positions):
        """Host torch tensors of the per-step decode inputs (positions: S ints, the position of the token being
        decoded in each slot; -1 = inactive slot: no cache write, open-pool state kept, attention output unused)."""
        S = self.S
        assert len(positions) == S
        active = torch.tensor([p >= 0 for p in positions])
        assert all(p < self.T for p in positions), f"decode position past max_context {self.T}"
        pos = torch.tensor([max(p, 0) for p in positions], dtype=torch.float32)
        phys = self._phys_rows(torch.arange(S), pos.long(), active)
        rows = torch.where(active, phys, torch.full((S,), 0xFFFFFFFF))
        j = pos.long() % KPOOL
        col = torch.arange(IDX_WIDTH).float().reshape(1, IDX_WIDTH)
        # inactive slots (pos -> 0) keep index 0 (row 0 of their own first block): every index row needs one valid entry
        # (sparse_sdpa gathers at least one key per row; an all-sentinel row would address before the cache)
        dense_idx = torch.where(col <= pos.reshape(S, 1), col, torch.full((S, IDX_WIDTH), SENTINEL_F))
        limit = torch.where(active, torch.floor((pos + 1) / KPOOL), torch.zeros(S))
        return {
            "row_idx": (rows.reshape(1, S).to(torch.int64), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            "pool_pos": (
                torch.where(active, pos.long() // KPOOL, torch.full((S,), -1)).to(torch.int32),
                ttnn.int32,
                ttnn.ROW_MAJOR_LAYOUT,
            ),
            "active": (active.float().reshape(1, 1, S, 1), ttnn.float32, ttnn.TILE_LAYOUT),
            "reset": ((j == 0).float().reshape(1, 1, S, 1), ttnn.float32, ttnn.TILE_LAYOUT),
            "ape_sel": (self.ape[j].reshape(1, 1, S, self.ID), ttnn.float32, ttnn.TILE_LAYOUT),
            "pos": (pos.reshape(1, 1, S, 1), ttnn.float32, ttnn.TILE_LAYOUT),
            "limit": (limit.reshape(1, 1, S, 1), ttnn.float32, ttnn.TILE_LAYOUT),
            "limit_u32": (limit.to(torch.int64).reshape(1, S), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            "dense": ((pos <= DENSE_MAX_POSITION).float().reshape(1, 1, S, 1), ttnn.float32, ttnn.TILE_LAYOUT),
            "dense_idx": (dense_idx.reshape(1, 1, S, IDX_WIDTH), ttnn.float32, ttnn.TILE_LAYOUT),
        }

    def decode_inputs(self, positions):
        """Allocate the fixed-address device buffers of the per-step decode inputs (before any trace capture)."""
        return {k: self._dev(t, dt, lay) for k, (t, dt, lay) in self._decode_host_inputs(positions).items()}

    def update_decode_inputs(self, bufs, positions):
        """Refresh the decode input buffers in place (host -> device copies; between trace replays)."""
        for k, (t, dt, lay) in self._decode_host_inputs(positions).items():
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(t, dtype=dt, layout=lay, mesh_mapper=self._rep), bufs[k])

    def decode_forward(self, x, state=None, *, inputs):
        """x [1,1,S,hidden] (one row per slot); inputs = decode_inputs(...) device tensors."""
        S = x.shape[-2]
        assert S == self.S
        q_abs, qr = self._q_abs(x, S)  # [1,H,S,512]
        c = self._latent(x)
        c_rm = ttnn.to_layout(c, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.scatter_rows(self.latent, ttnn.reshape(c_rm, (S, self.R)), inputs["row_idx"])
        ttnn.deallocate(c_rm)
        ttnn.deallocate(c)

        # ---- open-pool online-softmax update, then write every slot's (partial) pool key
        k, gl = self._index_k(x)  # [1,1,S,128]
        lg = ttnn.add(gl, inputs["ape_sel"])
        keep = ttnn.rsub(inputs["reset"], 1.0)  # 0 at a pool start
        m_prev = ttnn.add(ttnn.multiply(self.acc_m, keep), ttnn.multiply(inputs["reset"], NEG))
        m_new = ttnn.maximum(m_prev, lg)
        a_old = ttnn.multiply(ttnn.exp(ttnn.subtract(m_prev, m_new)), keep)
        a_new = ttnn.exp(ttnn.subtract(lg, m_new))
        den = ttnn.add(ttnn.multiply(self.acc_den, a_old), a_new)
        num = ttnn.add(ttnn.multiply(self.acc_num, a_old), ttnn.multiply(a_new, ttnn.typecast(k, ttnn.float32)))
        poolK = ttnn.typecast(ttnn.divide(num, den), ttnn.bfloat16)  # [1,1,S,128]
        # inactive slots keep their open-pool state (their pool key write is skipped: pool_pos -1)
        act = inputs["active"]
        for new, buf in ((m_new, self.acc_m), (den, self.acc_den), (num, self.acc_num)):
            kept = ttnn.where(act, new, buf)
            ttnn.copy(kept, buf)
            ttnn.deallocate(kept)
        poolK = ttnn.permute(poolK, (0, 2, 1, 3))  # [1,S,1,128]
        grid = self.mesh.compute_with_storage_grid_size()
        shard = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.ID),
            core_grid=ttnn.num_cores_to_corerangeset(S, grid, row_wise=True),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        pk_sh = ttnn.to_memory_config(poolK, shard)
        ttnn.experimental.paged_update_cache(
            self.pooled, pk_sh, update_idxs_tensor=inputs["pool_pos"], page_table=self.page_table
        )
        for t in (pk_sh, poolK, lg, keep, m_prev, m_new, a_old, a_new, den, num, k, gl):
            ttnn.deallocate(t)

        if self.T <= TOPK_POOLS * KPOOL:  # every position of this cache is in the dense regime
            o_lat = self._sparse_attend(q_abs, inputs["dense_idx"])
            ttnn.deallocate(q_abs)
            ttnn.deallocate(qr)
            y = self._out(o_lat, S)
            ttnn.deallocate(o_lat)
            return y
        # ---- selection: paged indexer scores of every slot over its own pools, masked, top-512
        qi = ttnn.reshape(self._mm(qr, self.w_iq), (1, 1, S * self.IH, self.ID))  # [1,1,S*32,128]
        wi = self._mm(x, self.w_iw, dtype=ttnn.float32)  # [1,1,S,32]
        sc = ttnn.experimental.dsa_decode_scores(
            qi, wi, self.pooled, self.page_table, inputs["limit_u32"], self.MB
        )  # [1,1,S,P_alloc] fp32, blocks past each slot's limit -inf
        ids = self._select_pools(sc, inputs["limit"])
        sparse = self._sparse_rows(ids, inputs["pos"], S)
        index = ttnn.where(inputs["dense"], inputs["dense_idx"], sparse)
        for t in (qi, wi, sc, ids, sparse):
            ttnn.deallocate(t)
        o_lat = self._sparse_attend(q_abs, index)
        ttnn.deallocate(index)
        ttnn.deallocate(q_abs)
        ttnn.deallocate(qr)
        y = self._out(o_lat, S)
        ttnn.deallocate(o_lat)
        return y


def _write_row(buf, src, idx: int):
    """buf [1,1,S,W] fp32 TILE: row idx <- src [1,1,1,W] (rebuild + copy; outside traces)."""
    S = buf.shape[-2]
    W = buf.shape[-1]
    parts = []
    if idx > 0:
        parts.append(ttnn.slice(buf, (0, 0, 0, 0), (1, 1, idx, W)))
    parts.append(src if src.dtype == buf.dtype else ttnn.typecast(src, buf.dtype))
    if idx < S - 1:
        parts.append(ttnn.slice(buf, (0, 0, idx + 1, 0), (1, 1, S, W)))
    new = ttnn.concat(parts, dim=-2)
    ttnn.copy(new, buf)
    ttnn.deallocate(new)
