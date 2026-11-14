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
                 weight_dtype=ttnn.bfloat8_b, kv_dtype=ttnn.bfloat8_b, own_global_cache: bool = True,
                 prefill_chunk: int = 2048):
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
        # explicit matmul precision (ttnn.linear's default config is low fidelity without fp32 accumulation): HiFi2 for
        # block-float weights (more mantissa than bfp8 holds), HiFi4 for bf16; fp32 accumulation
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4 if weight_dtype == ttnn.bfloat16 else ttnn.MathFidelity.HiFi2,
            math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
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
        self.prefill_chunk = prefill_chunk
        if self.sliding:
            # the ring holds the window + one prefill chunk + a block: a chunk's attention launch needs every row's
            # window live at once (decode alone would need only W + one block)
            self.n_ring = (self.W + prefill_chunk) // block_size + 1
            self.k_cache, self.v_cache = self._alloc_cache(self.n_ring)
            ring = (torch.arange(self.max_blocks) % self.n_ring).to(torch.int32).reshape(1, -1)
            self.ring_table = ttnn.from_torch(  # read side: logical block b -> physical b mod n_ring (the SDPA op)
                ring, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=DRAM,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            # write side: the stock paged_update_cache wants a table no wider than the cache, so writes go through an
            # identity table at position pos mod (n_ring * block) -- the same physical slot as the read side's mapping
            self.ring_period = self.n_ring * block_size
            self.ring_write_table = ttnn.from_torch(
                torch.arange(self.n_ring, dtype=torch.int32).reshape(1, -1), dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=DRAM,
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
        p = ttnn.linear(x, self.w_qkvr, compute_kernel_config=self.ckc, memory_config=L1)  # [1, 1, 1, fd]
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
        wpos, wpt = self._write_target(cur_pos, page_table)
        for cache, t in ((self.k_cache, k), (self.v_cache, v)):
            tp = ttnn.pad(t, [1, 1, 32, HD], [0, 0, 0, 0], 0.0, memory_config=L1)
            tsh = ttnn.to_memory_config(tp, self._kv_shard)
            ttnn.experimental.paged_update_cache(cache, tsh, update_idxs_tensor=wpos, page_table=wpt)
            ttnn.deallocate(tsh)
            ttnn.deallocate(tp)
        if wpos is not cur_pos:
            ttnn.deallocate(wpos)
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
        part = ttnn.linear(o, self.w_o, compute_kernel_config=self.ckc, memory_config=DRAM)  # [1, 1, 1, H] partial
        ttnn.deallocate(o)
        if not reduce or self.N == 1:
            return part
        from models.tt_transformers.tt.ccl import tt_all_reduce

        return tt_all_reduce(part, self.mesh, self.tt_ccl, cluster_axis=0, dim=3, topology=ttnn.Topology.Linear,
                             memory_config=DRAM)

    def forward_rows(self, x, positions, page_table, tau=None, commit_all=True, reduce=True):
        """R <= 8 rows at consecutive positions (spec verify / prefill in row groups): x [1, 1, R, H] replicated (normed),
        positions [R] int32 (the rows' positions). Each row attends causally to its own position; every row's K/V is
        written (a rejected verify row is overwritten by the next pass). The k/v sconv states advance past all R rows
        (commit_all) or wait for commit_rows(n). -> [1, 1, R, H/N] fractured (reduce) or the [1, 1, R, H] partial."""
        NH, NKV, HD, DR = self.NH, self.NKV, self.HD, self.DREL
        R = int(x.shape[2])
        assert R * (NH // NKV) <= 32, f"{R} rows x {NH // NKV} q heads per kv head must fit one tile"
        p = ttnn.linear(x, self.w_qkvr, compute_kernel_config=self.ckc, memory_config=L1)  # [1, 1, R, fd]
        qd, kd, rd = self.qd, self.kd, self.rd
        q = ttnn.slice(p, (0, 0, 0, 0), (1, 1, R, qd), memory_config=L1)
        k = ttnn.slice(p, (0, 0, 0, qd), (1, 1, R, qd + kd), memory_config=L1)
        v = ttnn.slice(p, (0, 0, 0, qd + kd), (1, 1, R, qd + 2 * kd), memory_config=L1)
        r = ttnn.slice(p, (0, 0, 0, qd + 2 * kd), (1, 1, R, self.fd), memory_config=L1)
        ttnn.deallocate(p)
        k = self.k_sconv.rows(k, commit_all=commit_all)
        v = self.v_sconv.rows(v, commit_all=commit_all)
        q = ttnn.reshape(q, (1, R, NH, HD))
        q = ttnn.multiply(ttnn.rms_norm(q, epsilon=1e-6, memory_config=L1), self.q_norm, memory_config=L1)
        k = ttnn.reshape(k, (1, R, NKV, HD))
        k = ttnn.multiply(ttnn.rms_norm(k, epsilon=1e-6, memory_config=L1), self.k_norm, memory_config=L1)
        v = ttnn.reshape(v, (1, R, NKV, HD))
        r = ttnn.reshape(r, (1, R, NH, DR))
        if tau is not None and self.log_scaling:  # tau [1, R, 1, 1]: per-row
            q = ttnn.multiply(q, tau, memory_config=L1)
            r = ttnn.multiply(r, tau, memory_config=L1)
        pt = self.ring_table if self.sliding else page_table
        for j in range(R):  # one row at a time: its own position
            # (with R == 1 a whole-tensor slice can return its input: never free a slice that aliases it)
            pos_j = positions if R == 1 else ttnn.slice(positions, (j,), (j + 1,))
            wpos, wpt = self._write_target(pos_j, page_table)
            for cache, t in ((self.k_cache, k), (self.v_cache, v)):
                row = t if R == 1 else ttnn.slice(t, (0, j, 0, 0), (1, j + 1, NKV, HD), memory_config=L1)
                tp = ttnn.pad(row, [1, 1, 32, HD], [0, 0, 0, 0], 0.0, memory_config=L1)
                tsh = ttnn.to_memory_config(tp, self._kv_shard)
                ttnn.experimental.paged_update_cache(cache, tsh, update_idxs_tensor=wpos, page_table=wpt)
                for z in (tsh, tp) + (() if row is t else (row,)):
                    ttnn.deallocate(z)
            if wpos is not pos_j:
                ttnn.deallocate(wpos)
            if pos_j is not positions:
                ttnn.deallocate(pos_j)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        q_in = ttnn.to_memory_config(q, DRAM)
        ttnn.deallocate(q)
        rq = ttnn.pad(r, [1, R, NH, 32], [0, 0, 0, 0], 0.0, memory_config=DRAM)
        ttnn.deallocate(r)
        pt_rows = self._rows_page_table(pt, R)
        o = inkling_sdpa_decode(q_in, self.k_cache, self.v_cache, positions, pt_rows, rq, self.table, self.scale,
                                self.W, self.E)
        ttnn.deallocate(q_in)
        ttnn.deallocate(rq)
        o = ttnn.reshape(o, (1, 1, R, NH * HD))
        part = ttnn.linear(o, self.w_o, compute_kernel_config=self.ckc, memory_config=DRAM)
        ttnn.deallocate(o)
        if not reduce or self.N == 1:
            return part
        from models.tt_transformers.tt.ccl import tt_all_reduce

        return tt_all_reduce(part, self.mesh, self.tt_ccl, cluster_axis=0, dim=3, topology=ttnn.Topology.Linear,
                             memory_config=DRAM)

    def forward_prefill(self, x, p0: int, page_table_host: torch.Tensor | None = None, reduce=True):
        """A prefill chunk in ONE attention launch: x [1, 1, T, H] replicated (normed) at positions p0 .. p0 + T - 1,
        p0 a multiple of the block size, T a multiple of 64 (pad the chunk; padded rows write KV past the prompt, which
        decode overwrites before any row can attend to it) and <= prefill_chunk. page_table_host [max_blocks] int32:
        the sequence's logical -> physical blocks (global layers). The k/v conv states advance past all T rows.
        -> [1, 1, T, H/N] fractured (reduce) or the [1, 1, T, H] partial."""
        NH, NKV, HD, DR = self.NH, self.NKV, self.HD, self.DREL
        T = int(x.shape[2])
        bs = self.block
        assert p0 % bs == 0 and T % 64 == 0 and T <= self.prefill_chunk, (p0, T, self.prefill_chunk)
        mc = DRAM
        p = ttnn.linear(x, self.w_qkvr, compute_kernel_config=self.ckc, memory_config=mc)  # [1, 1, T, fd]
        qd, kd, rd = self.qd, self.kd, self.rd
        q = ttnn.slice(p, (0, 0, 0, 0), (1, 1, T, qd), memory_config=mc)
        k = ttnn.slice(p, (0, 0, 0, qd), (1, 1, T, qd + kd), memory_config=mc)
        v = ttnn.slice(p, (0, 0, 0, qd + kd), (1, 1, T, qd + 2 * kd), memory_config=mc)
        r = ttnn.slice(p, (0, 0, 0, qd + 2 * kd), (1, 1, T, self.fd), memory_config=mc)
        ttnn.deallocate(p)
        k = self.k_sconv.rows(k, commit_all=True)
        v = self.v_sconv.rows(v, commit_all=True)
        q = ttnn.reshape(q, (1, T, NH, HD))
        q = ttnn.multiply(ttnn.rms_norm(q, epsilon=1e-6, memory_config=mc), self.q_norm, memory_config=mc)
        k = ttnn.reshape(k, (1, T, NKV, HD))
        k = ttnn.multiply(ttnn.rms_norm(k, epsilon=1e-6, memory_config=mc), self.k_norm, memory_config=mc)
        v = ttnn.reshape(v, (1, T, NKV, HD))
        r = ttnn.reshape(r, (1, T, NH, DR))
        if self.log_scaling and p0 + T > 128000:  # tau per row (exact host values; a no-op below 128K)
            pos = torch.arange(p0, p0 + T, dtype=torch.float64) + 1.0
            tau = (1.0 + 0.1 * torch.log(torch.clamp(pos / 128000.0, min=1.0))).float().reshape(1, T, 1, 1)
            tau_t = ttnn.from_torch(tau, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))
            q = ttnn.multiply(q, tau_t, memory_config=mc)
            r = ttnn.multiply(r, tau_t, memory_config=mc)
            ttnn.deallocate(tau_t)
        # KV fill of the chunk: [1, NKV, T, HD] in the cache dtype, through this chunk's blocks
        b0, nb = p0 // bs, T // bs
        if self.sliding:
            blocks = (torch.arange(b0, b0 + nb) % self.n_ring).to(torch.int32)
        else:
            blocks = page_table_host.reshape(-1)[b0 : b0 + nb].to(torch.int32)
        fill_pt = ttnn.from_torch(blocks.reshape(1, -1), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                  device=self.mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))
        for cache, t in ((self.k_cache, k), (self.v_cache, v)):
            tt_ = ttnn.transpose(t, 1, 2, memory_config=mc)  # [1, NKV, T, HD]
            t8 = ttnn.typecast(tt_, self.kv_dtype) if self.kv_dtype != tt_.dtype else tt_
            ttnn.experimental.paged_fill_cache(cache, t8, fill_pt, batch_idx=0)
            if t8 is not tt_:
                ttnn.deallocate(t8)
            ttnn.deallocate(tt_)
        ttnn.deallocate(fill_pt)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        rq = ttnn.pad(r, [1, T, NH, 32], [0, 0, 0, 0], 0.0, memory_config=mc)
        ttnn.deallocate(r)
        positions = ttnn.from_torch(torch.arange(p0, p0 + T, dtype=torch.int32).reshape(-1, 8), dtype=ttnn.int32,
                                    layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh,
                                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))
        if self.sliding:
            pt = self.ring_table
        else:
            pt = ttnn.from_torch(page_table_host.reshape(1, -1).to(torch.int32), dtype=ttnn.int32,
                                 layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh,
                                 mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))
        o = inkling_sdpa_decode(q, self.k_cache, self.v_cache, positions, pt, rq, self.table, self.scale, self.W,
                                self.E)
        for z in (q, rq, positions):
            ttnn.deallocate(z)
        if not self.sliding:
            ttnn.deallocate(pt)
        o = ttnn.reshape(o, (1, 1, T, NH * HD))
        part = ttnn.linear(o, self.w_o, compute_kernel_config=self.ckc, memory_config=DRAM)
        ttnn.deallocate(o)
        if not reduce or self.N == 1:
            return part
        from models.tt_transformers.tt.ccl import tt_all_reduce

        return tt_all_reduce(part, self.mesh, self.tt_ccl, cluster_axis=0, dim=3, topology=ttnn.Topology.Linear,
                             memory_config=DRAM)

    def commit_rows(self, n: int):
        """Spec verify: the k/v conv states advance past the first n of the last forward_rows' rows."""
        self.k_sconv.commit_rows(n)
        self.v_sconv.commit_rows(n)

    def _write_target(self, pos, page_table):
        """(update index tensor, page table) for paged_update_cache at positions `pos` (int32 row-major)."""
        if not self.sliding:
            return pos, page_table
        return ttnn.remainder(pos, self.ring_period), self.ring_write_table

    def _rows_page_table(self, pt, R):
        """The SDPA op reads page-table row 0 for every row (one sequence): a [1, max_blocks] table serves any R."""
        return pt
