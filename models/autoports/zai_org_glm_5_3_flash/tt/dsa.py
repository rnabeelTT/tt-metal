# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash DeepSeek sparse attention (DSA) layer: NoPE MLA over a 512-wide latent + lightning indexer.

HF (`Glm5NextTextAttention` + `Glm5NextTextIndexer`), 64 heads, no positional encoding anywhere:
    q_r  = rmsnorm(q_a(x))                          [T, 1536] (shared with the indexer)
    q    = q_b(q_r)  -> [T, 64, 256]
    c    = rmsnorm(kv_a(x))                         [T, 512]  (the cached latent)
    k_h  = W_UK_h c,  v_h = W_UV_h c                (kv_b rows h*512 + [0,256) / [256,512))
    o_h  = softmax(q_h k_h^T / 16 + M) v_h ;  out = o_proj(concat_h o_h)
with M the sparse mask from the indexer: per query t the top-512 complete 4-token pools by
    I[t,p] = sum_h w[t,h] * relu(qI[t,h] . poolK[p] / sqrt(128))   (w = weights_proj(x) / sqrt(32))
plus the raw tail of the query's incomplete pool. When a query sees <= 512 complete pools (position t <= 2050)
the selection is every visible key: exact dense causal attention.

Absorbed form used here (identical math): q_abs_h = W_UK_h^T q_h [512]; attention of q_abs over the latent rows
(MQA, K = V = latent, scale 1/16); o_h = W_UV_h o_lat. Caches (per layer, paged, block_size rows per block):
    latent  [num_blocks, 1, block_size, 512]       bf16 TILE
The indexer pooled-key cache and the sparse path are added in this file below the dense path.
"""

from __future__ import annotations

import math

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

DRAM = ttnn.DRAM_MEMORY_CONFIG
DENSE_MAX_POSITION = 2050  # query positions <= this attend to every visible key (HF selection is exhaustive)


class GlmDSA(LightweightModule):
    def __init__(
        self,
        mesh_device,
        tc: dict,
        sd: dict[str, torch.Tensor],
        layer_idx: int,
        *,
        max_batch: int = 32,
        num_blocks: int = 64,
        block_size: int = 64,
        weight_dtype=ttnn.bfloat8_b,
    ):
        self.mesh = mesh_device
        self.layer_idx = layer_idx
        self.tp = tuple(mesh_device.shape)[1]
        assert self.tp == 1, "functional DSA: single device (TP is added in the multichip stage)"
        self.H = tc["num_attention_heads"]
        self.Dn = tc["qk_nope_head_dim"]  # 256
        self.Dv = tc["v_head_dim"]  # 256
        self.R = tc["kv_lora_rank"]  # 512
        self.QR = tc["q_lora_rank"]  # 1536
        self.hidden = tc["hidden_size"]
        self.eps = tc["rms_norm_eps"]
        self.scale = self.Dn**-0.5  # qk_head_dim = 256 (nope only) -> 1/16
        self.max_batch = max_batch
        self.num_blocks = num_blocks
        self.block_size = block_size
        rep = ttnn.ReplicateTensorToMesh(mesh_device)

        def w(t, dtype=weight_dtype):
            return ttnn.from_torch(
                t.contiguous(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=DRAM,
                mesh_mapper=rep,
            )

        def norm_w(t):
            return ttnn.from_torch(
                t.to(torch.bfloat16).reshape(1, 1, -1, 32),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                mesh_mapper=rep,
            )

        a = "self_attn."
        self.w_q_a = w(sd[a + "q_a_proj.weight"].T)
        self.q_a_norm = norm_w(sd[a + "q_a_layernorm.weight"])
        self.w_q_b = w(sd[a + "q_b_proj.weight"].T)  # [1536, 64*256]
        self.w_kv_a = w(sd[a + "kv_a_proj_with_mqa.weight"].T)  # [4096, 512]
        self.kv_a_norm = norm_w(sd[a + "kv_a_layernorm.weight"])
        kvb = sd[a + "kv_b_proj.weight"].float().reshape(self.H, self.Dn + self.Dv, self.R)
        w_uk = kvb[:, : self.Dn, :]  # [H, 256, 512]: k_h = W_UK_h c
        w_uv = kvb[:, self.Dn :, :]  # [H, 256, 512]: v_h = W_UV_h c
        # q_abs_h = q_h @ W_UK_h  ([.., 256] @ [256, 512]);  o_h = o_lat @ W_UV_h^T  ([.., 512] @ [512, 256])
        self.w_uk = w(w_uk.reshape(1, self.H, self.Dn, self.R), ttnn.bfloat16)
        self.w_uv_t = w(w_uv.transpose(1, 2).reshape(1, self.H, self.R, self.Dv), ttnn.bfloat16)
        self.w_o = w(sd[a + "o_proj.weight"].T)  # [64*256, 4096]
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.sdpa_ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        grid = mesh_device.compute_with_storage_grid_size()
        # decode: 64 heads x 512 latent per user; k chunk 32 keeps the circular buffers inside L1
        self.decode_progcfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=0, k_chunk_size=32, exp_approx_mode=False
        )
        self.latent = None

    # ----------------------------------------------------------------------------------------------- caches
    def allocate_decode_state(self, batch: int | None = None):
        """The paged latent cache (shared by prefill and decode); `batch` only bounds the decode width."""
        if self.latent is None:
            self.latent = ttnn.zeros(
                (self.num_blocks, 1, self.block_size, self.R),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=DRAM,
            )
        return self.latent

    # ------------------------------------------------------------------------------------------ projections
    def _rms(self, x, w):
        return ttnn.rms_norm(x, epsilon=self.eps, weight=w, memory_config=DRAM)

    def _q_abs(self, x, rows: int):
        """x [1,1,rows,hidden] -> q_abs [1, H, rows, 512]."""
        qr = self._rms(ttnn.linear(x, self.w_q_a, compute_kernel_config=self.ckc, memory_config=DRAM), self.q_a_norm)
        q = ttnn.linear(qr, self.w_q_b, compute_kernel_config=self.ckc, memory_config=DRAM)  # [1,1,rows,H*256]
        q = ttnn.reshape(q, (1, rows, self.H, self.Dn))
        q = ttnn.permute(q, (0, 2, 1, 3), memory_config=DRAM)  # [1,H,rows,256]
        q_abs = ttnn.matmul(q, self.w_uk, compute_kernel_config=self.ckc, memory_config=DRAM)  # [1,H,rows,512]
        ttnn.deallocate(q)
        return q_abs, qr

    def _latent(self, x):
        return self._rms(
            ttnn.linear(x, self.w_kv_a, compute_kernel_config=self.ckc, memory_config=DRAM), self.kv_a_norm
        )

    def _out(self, o_lat, rows: int):
        """o_lat [1, H, rows, 512] -> [1,1,rows,hidden]."""
        o = ttnn.matmul(o_lat, self.w_uv_t, compute_kernel_config=self.ckc, memory_config=DRAM)  # [1,H,rows,256]
        o = ttnn.permute(o, (0, 2, 1, 3), memory_config=DRAM)  # [1,rows,H,256]
        o = ttnn.reshape(o, (1, 1, rows, self.H * self.Dv))
        return ttnn.linear(o, self.w_o, compute_kernel_config=self.ckc, memory_config=DRAM)

    # ---------------------------------------------------------------------------------------------- prefill
    def prefill_forward(self, x, *, start: int, length: int, page_table, user: int = 0):
        """One user's chunk x [1,1,n,hidden] at absolute positions [start, start+n) (start % 32 == 0; rows >= length
        are padding). page_table: device int32 [1, max_blocks] of this user (row `user` of the batch table is not
        needed: one user per call). Writes the chunk's latent into the paged cache. Returns (y [1,1,n,hidden], None)."""
        n = x.shape[-2]
        if start + length - 1 > DENSE_MAX_POSITION:
            raise NotImplementedError("sparse DSA prefill path")  # filled in below the dense path
        q_abs, qr = self._q_abs(x, n)
        ttnn.deallocate(qr)
        c = self._latent(x)  # [1,1,n,512]
        # cache fill: the chunk's page table starts at the chunk's first block
        first_block = start // self.block_size
        nblk = math.ceil(n / self.block_size)
        chunk_pt = ttnn.slice(page_table, (0, first_block), (1, first_block + nblk))
        ttnn.experimental.paged_fill_cache(self.latent, c, page_table=chunk_pt, batch_idx=0)
        if start == 0:
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
            o_lat = ttnn.transformer.chunked_flash_mla_prefill(
                q_abs,
                self.latent,
                self.R,
                page_table,
                start,
                scale=self.scale,
                compute_kernel_config=self.sdpa_ckc,
                memory_config=DRAM,
            )
        ttnn.deallocate(q_abs)
        ttnn.deallocate(c)
        ttnn.deallocate(chunk_pt)
        y = self._out(o_lat, n)
        ttnn.deallocate(o_lat)
        return y, None

    # ----------------------------------------------------------------------------------------------- decode
    def decode_forward(self, x, latent, *, cur_pos, page_table):
        """x [1,1,B,hidden]; cur_pos: device int32 [B] (position of the token being decoded, -1 = inactive);
        page_table: device int32 [B, max_blocks]. Appends the latent row and attends over positions <= cur_pos."""
        B = x.shape[-2]
        q_abs, qr = self._q_abs(x, B)  # [1,H,B,512]
        ttnn.deallocate(qr)
        c = self._latent(x)  # [1,1,B,512]
        c = ttnn.permute(c, (0, 2, 1, 3), memory_config=DRAM)  # [1,B,1,512]
        # paged_update_cache takes the update height-sharded: one user (one 32-row tile) per core
        grid = self.mesh.compute_with_storage_grid_size()
        shard = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.R),
            core_grid=ttnn.num_cores_to_corerangeset(B, grid, row_wise=True),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        c_sh = ttnn.to_memory_config(c, shard)
        ttnn.deallocate(c)
        ttnn.experimental.paged_update_cache(self.latent, c_sh, update_idxs_tensor=cur_pos, page_table=page_table)
        ttnn.deallocate(c_sh)
        q = ttnn.permute(q_abs, (0, 2, 1, 3), memory_config=DRAM)  # [1,B,H,512]
        ttnn.deallocate(q_abs)
        o_lat = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            q,
            self.latent,
            None,
            self.R,
            page_table,
            cur_pos_tensor=cur_pos,
            scale=self.scale,
            program_config=self.decode_progcfg,
            compute_kernel_config=self.sdpa_ckc,
            memory_config=DRAM,
        )  # [1,B,H,512]
        ttnn.deallocate(q)
        o_lat = ttnn.permute(o_lat, (0, 2, 1, 3), memory_config=DRAM)  # [1,H,B,512]
        y = self._out(o_lat, B)
        ttnn.deallocate(o_lat)
        return y
