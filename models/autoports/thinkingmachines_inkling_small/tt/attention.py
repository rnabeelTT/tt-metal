# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Inkling attention, tensor-parallel over a (1, N) mesh (N = 4 on a QB2): 32 q heads / 8 kv heads -> 8 / 2 per die.

Per layer (transformers `InklingAttention`):
    q, k, v, r = x @ Wq, x @ Wk, x @ Wv, x @ Wr          (r: 16 per head, the relative-bias mixing weights)
    k, v = k_sconv(k), v_sconv(v)                        (fp32 short convs, tt/sconv.py)
    q, k = rms_norm(q) * q_norm, rms_norm(k) * k_norm   (per head; plain weight, no +1)
    global layers past 128K: q, r *= tau(pos)
    o = softmax(q k^T / head_dim + bias(r, proj)) v      (tt/attn/inkling_sdpa.py; sliding layers: window 512)
    out = o @ Wo   -> reduce-scatter                     (the layer applies attn_sconv to its fractured slice)

Weights per die: one fused column-parallel [q_d | k_d | v_d | r_d] projection (die d: q heads 8d..8d+7, kv heads 2d,
2d+1, r rows of q heads 8d..8d+7), row-parallel Wo, replicated q/k norms and bias table, k/v sconv taps on the die's
kv-head channels.

KV caches: global layers read/write a paged cache (the serving engine's, or one allocated here) through the request's
page table; sliding layers own a ring of n_ring = W / block + 1 blocks addressed through a fixed page table
(logical block b -> b mod n_ring), so they hold W + one block of KV whatever the context length.

Implemented: one-row decode (correctness first; the fused/optimised paths come later). Prefill / spec-verify rows land
here next.
"""

from __future__ import annotations

import math

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_small.tt import relbias
from models.autoports.thinkingmachines_inkling_small.tt.attn.inkling_sdpa import inkling_sdpa_decode
from models.autoports.thinkingmachines_inkling_small.tt.sconv import ShortConv

L1 = ttnn.L1_MEMORY_CONFIG
DRAM = ttnn.DRAM_MEMORY_CONFIG


class InklingAttention:
    def __init__(self, mesh, cfg, sd: dict, layer_idx: int, tt_ccl=None, block_size: int = 64, max_seq_len: int = 32768,
                 weight_dtype=ttnn.bfloat8_b, kv_dtype=ttnn.bfloat8_b, own_global_cache: bool = True):
        """cfg: the HF text config (InklingTextConfig or a namespace with the same fields). sd: this layer's attention
        weights under the HF module names (q_proj.weight, ..., k_sconv.conv1d.weight, rel_logits_proj.proj)."""
        self.mesh = mesh
        self.tt_ccl = tt_ccl
        self.layer_idx = layer_idx
        self.N = mesh.get_num_devices()
        self.sliding = cfg.layer_types[layer_idx] == "hybrid_sliding"
        self.HD = cfg.swa_head_dim if self.sliding else cfg.head_dim
        nh = cfg.swa_num_attention_heads if self.sliding else cfg.num_attention_heads
        nkv = cfg.swa_num_key_value_heads if self.sliding else cfg.num_key_value_heads
        assert nh % self.N == 0 and nkv % self.N == 0, (nh, nkv, self.N)
        self.NH, self.NKV = nh // self.N, nkv // self.N
        self.DREL = cfg.d_rel
        self.W = cfg.sliding_window_size if self.sliding else 0
        self.E = cfg.sliding_window_size if self.sliding else cfg.rel_extent
        self.scale = 1.0 / self.HD
        self.log_scaling = (not self.sliding) and cfg.log_scaling_n_floor is not None
        self.block = block_size
        self.kv_dtype = kv_dtype
        H, HD, NH, NKV, DR, N = cfg.hidden_size, self.HD, self.NH, self.NKV, self.DREL, self.N

        # fused [q_d | k_d | v_d | r_d] per die, die-major rows -> column-sharded
        wq, wk, wv, wr = (sd[f"{n}_proj.weight"].to(torch.bfloat16) for n in ("q", "k", "v", "r"))
        blocks = []
        for d in range(N):
            blocks += [
                wq[d * NH * HD : (d + 1) * NH * HD],
                wk[d * NKV * HD : (d + 1) * NKV * HD],
                wv[d * NKV * HD : (d + 1) * NKV * HD],
                wr[d * NH * DR : (d + 1) * NH * DR],
            ]
        fused = torch.cat(blocks, 0)  # [N * (NH*HD + 2*NKV*HD + NH*DR), H]
        self.qd, self.kd, self.rd = NH * HD, NKV * HD, NH * DR
        self.fd = self.qd + 2 * self.kd + self.rd
        shard_cols = lambda w: ttnn.ShardTensorToMesh(mesh, dim=-1)  # noqa: E731
        self.w_qkvr = ttnn.as_tensor(
            fused.T.contiguous(), dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=mesh,
            memory_config=DRAM, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
        )
        self.w_o = ttnn.as_tensor(
            sd["o_proj.weight"].to(torch.bfloat16).T.contiguous(), dtype=weight_dtype, layout=ttnn.TILE_LAYOUT,
            device=mesh, memory_config=DRAM, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-2),
        )
        rep = lambda t: ttnn.as_tensor(  # noqa: E731
            t.reshape(1, 1, 1, -1).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
            memory_config=DRAM, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.q_norm = rep(sd["q_norm.weight"])
        self.k_norm = rep(sd["k_norm.weight"])
        kv_shard = ttnn.ShardTensorToMesh(mesh, dim=-1)
        self.k_sconv = ShortConv(mesh, sd["k_sconv.conv1d.weight"].reshape(nkv * HD, -1), kv_shard, NKV * HD)
        self.v_sconv = ShortConv(mesh, sd["v_sconv.conv1d.weight"].reshape(nkv * HD, -1), kv_shard, NKV * HD)
        T = relbias.build_tables(sd["rel_logits_proj.proj"].float(), self.scale)
        self.table = ttnn.as_tensor(
            T.reshape(1, 32, 32, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
            memory_config=DRAM, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        del shard_cols

        # KV caches
        self.max_blocks = math.ceil(max_seq_len / block_size)
        if self.sliding:
            self.n_ring = self.W // block_size + 1
            self.k_cache, self.v_cache = self._alloc_cache(self.n_ring)
            ring = (torch.arange(self.max_blocks) % self.n_ring).to(torch.int32).reshape(1, -1)
            self.ring_table = ttnn.from_torch(
                ring, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=DRAM,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
        elif own_global_cache:
            self.k_cache, self.v_cache = self._alloc_cache(self.max_blocks)
        else:
            self.k_cache = self.v_cache = None  # bound by the serving engine (bind_cache)
        self._kv_shard = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, HD), core_grid=ttnn.CoreGrid(x=1, y=1), strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True,
        )

    def _alloc_cache(self, n_blocks):
        """Zeroed [n_blocks, NKV, block, HD] per die (zeros, not garbage: masked positions are read and must be finite)."""
        z = torch.zeros(n_blocks, self.NKV * self.N, self.block, self.HD)
        mk = lambda: ttnn.from_torch(  # noqa: E731
            z, dtype=self.kv_dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=DRAM,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=1),
        )
        return mk(), mk()

    def bind_cache(self, k_cache, v_cache):
        assert not self.sliding
        self.k_cache, self.v_cache = k_cache, v_cache

    def reset(self):
        """New sequence: zero the conv states (the ring cache needs no reset: positions are masked by the window)."""
        self.k_sconv.reset()
        self.v_sconv.reset()

    def forward_decode(self, x, cur_pos, page_table, tau=None, reduce=True):
        """x [1, 1, 1, H] replicated (normed); cur_pos [1] int32; page_table [1, max_blocks] int32 (global layers);
        tau [1, 1, 1, 1] (global layers past 128K, else None). -> [1, 1, 1, H/N] fractured (reduce) or the [1, 1, 1, H]
        partial sum."""
        NH, NKV, HD, DR = self.NH, self.NKV, self.HD, self.DREL
        p = ttnn.linear(x, self.w_qkvr, memory_config=L1)  # [1, 1, 1, fd]
        qd, kd, rd = self.qd, self.kd, self.rd
        q = ttnn.slice(p, (0, 0, 0, 0), (1, 1, 1, qd), memory_config=L1)
        k = ttnn.slice(p, (0, 0, 0, qd), (1, 1, 1, qd + kd), memory_config=L1)
        v = ttnn.slice(p, (0, 0, 0, qd + kd), (1, 1, 1, qd + 2 * kd), memory_config=L1)
        r = ttnn.slice(p, (0, 0, 0, qd + 2 * kd), (1, 1, 1, self.fd), memory_config=L1)
        ttnn.deallocate(p)
        k = self.k_sconv.decode(k)
        v = self.v_sconv.decode(v)
        q = ttnn.reshape(q, (1, 1, NH, HD))
        q = ttnn.multiply(ttnn.rms_norm(q, epsilon=1e-6, memory_config=L1), self.q_norm, memory_config=L1)
        k = ttnn.reshape(k, (1, 1, NKV, HD))
        k = ttnn.multiply(ttnn.rms_norm(k, epsilon=1e-6, memory_config=L1), self.k_norm, memory_config=L1)
        v = ttnn.reshape(v, (1, 1, NKV, HD))
        if tau is not None and self.log_scaling:
            q = ttnn.multiply(q, tau, memory_config=L1)
            r = ttnn.multiply(r, tau, memory_config=L1)
        # KV write at cur_pos
        pt = self.ring_table if self.sliding else page_table
        for cache, t in ((self.k_cache, k), (self.v_cache, v)):
            tp = ttnn.pad(t, [1, 1, 32, HD], [0, 0, 0, 0], 0.0, memory_config=L1)
            tsh = ttnn.to_memory_config(tp, self._kv_shard)
            ttnn.experimental.paged_update_cache(cache, tsh, update_idxs_tensor=cur_pos, page_table=pt)
            ttnn.deallocate(tsh)
            ttnn.deallocate(tp)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        # the op's inputs: q [1, R=1, NH, HD] and rq [1, 1, NH, 32] interleaved DRAM
        q_in = ttnn.to_memory_config(q, DRAM)
        ttnn.deallocate(q)
        r = ttnn.reshape(r, (1, 1, NH, DR))
        rq = ttnn.pad(r, [1, 1, NH, 32], [0, 0, 0, 0], 0.0, memory_config=DRAM)
        ttnn.deallocate(r)
        o = inkling_sdpa_decode(q_in, self.k_cache, self.v_cache, cur_pos, pt, rq, self.table, self.scale, self.W, self.E)
        ttnn.deallocate(q_in)
        ttnn.deallocate(rq)
        o = ttnn.reshape(o, (1, 1, 1, NH * HD))
        part = ttnn.linear(o, self.w_o, memory_config=DRAM)  # [1, 1, 1, H] partial
        ttnn.deallocate(o)
        if not reduce or self.N == 1:
            return part
        from models.tt_transformers.tt.ccl import tt_all_reduce

        return tt_all_reduce(part, self.mesh, self.tt_ccl, cluster_axis=0, dim=3, topology=ttnn.Topology.Linear,
                             memory_config=DRAM)
