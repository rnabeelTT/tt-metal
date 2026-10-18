# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Block-sharded DSA latent cache for the TP decoder (MCOpts.dsa_shard, or GLM53_DSA_SHARD=1): the paged latent pool is
spread over the tp dies instead of replicated, so the long-context capacity is not capped by the replicated latent.

Layout (the paged-pool API is unchanged: page tables / vLLM block tables hold GLOBAL block ids):
  * global block b (b < NB - 1; NB - 1 is GlmDSA's dummy block) lives on die b % tp, at local block b // tp of the die's
    pool [1, 1, (NBL + 2) * 128, 544] bf16 ROW_MAJOR, NBL = ceil((NB - 1) / tp). The per-die rows are 544 wide: the
    512-column latent and 32 control columns (below). Local block NBL is the NULL block, NBL + 1 the SINK block.
  * the pooled index keys, the indexer, the top-512 pool selection and the open-pool accumulators stay REPLICATED (pool
    keys are 64 B / token / layer against 1024 B of latent): the selected index list is identical on every die and
    to the replicated layout. A token's latent row is written only on its owner die (per-die scatter row index).
  * each die maps the (replicated) logical index list through its own page table lpt [S, MB + 1]: a block it owns ->
    its local block, any other block -> NULL; column MB - 1 (the indexer overrun column, never a real position) ->
    SINK, column MB -> NULL. Every index list gets a 32-entry prefix [sink, 31 x null].

Control columns of a cache row / of the query (q_abs extended to 544):
            col 512 (ONE)   col 513 (SINK)   col 514 (KILL)
  real row      1               0                0           latent of a written token
  sink row      0               1                0           the whole row is otherwise zero
  null row      0               0             -1e4           (and a zero latent)
  query         0           q.c_self            1           c_self = the query token's own latent row
so a real key scores q.k as before, the sink scores q.c_self (the query's own-key logit: the same value on every die,
inside the attended logit range), a null key scores -1e4 (weight exactly 0 after exp), and the output columns are
  o[:512] = sum_local p v / D,  o[512] = u = sum_local p / D,  o[513] = w = p_sink / D   (D: the die's softmax sum).

Decode / prefill attention per DSA layer (rows = query rows):
  1. q_ext [1, 16, rows, 544] (the die's heads) -> all-gather -> [1, 64, rows, 544]: every die attends every head over
     its own keys (sparse_sdpa, H = 64, v_dim 544, the die's pool and page table).
  2. per die, in the common unit p_sink = exp(sigma - max) (sigma identical on every die): the unnormalized numerator
     T = o / w (columns :512 = sum_local p v / p_sink) and the softmax mass M = m / w (sum_local p / p_sink), with
     m = 1 - w for w < 0.5 (u + w = 1 up to rounding: no bf16 cancellation) and m = u otherwise.
  3. reduce-scatter (sum, fp32) of T and of M over the head groups -> the die's 16 heads: o_lat = sum T / sum M.
That is the flash-decode LSE merge without an LSE output from the kernel. Prefill rows run in blocks of
GLM53_DSA_SHARD_ROWS (default 256) to bound the gathered q / partial buffers.
"""

from __future__ import annotations

import os

import torch

import ttnn

from .dsa import BLOCK, IDX_WIDTH, SENTINEL_F

DRAM = ttnn.DRAM_MEMORY_CONFIG
KCOL = 544  # per-die cache row: 512 latent + 32 control columns
C_ONE, C_SINK, C_KILL = 512, 513, 514
KILL = 1.0e4
PREFIX = 32  # [sink, 31 x null] in front of every index list
TINY = 1.0e-30


def shard_rows_default() -> int:
    return int(os.environ.get("GLM53_DSA_SHARD_ROWS", "256"))


class ShardedDSAMixin:
    """Mixed in front of the decoder's DSA class (install_dsa_shard). Only the latent pool, its page table, the latent
    writes and the sparse attention change; everything else is the replicated GlmDSA."""

    dsa_shard = True

    def shard_init(self, ccl):
        assert self.tp > 1 and self.tp & (self.tp - 1) == 0, f"block sharding needs a power-of-two tp > 1 ({self.tp})"
        assert self.latent is None, "install the sharded cache before allocate_decode_state"
        self.sccl = ccl
        self.shard_rows = shard_rows_default()
        # numerator reduce-scatter payload: fp32 (default) or bf16 (GLM53_DSA_SHARD_RS=bf16: half the prefill CCL bytes)
        self.shard_rs_dtype = ttnn.bfloat16 if os.environ.get("GLM53_DSA_SHARD_RS", "fp32") == "bf16" else ttnn.float32
        self.lpt = None
        self._dies_map = ttnn.ShardTensorToMesh(self.mesh, dim=0)

    # ------------------------------------------------------------------------------------------- helpers
    def _dies(self, t, dtype, layout, host=False):
        """t: the per-die tensors concatenated along dim 0 (die r = r-th slice) -> one mesh tensor."""
        kw = {} if host else {"device": self.mesh}
        return ttnn.from_torch(t, dtype=dtype, layout=layout, mesh_mapper=self._dies_map, **kw)

    def dsa_bytes_per_block(self) -> float:
        """DRAM per die per 128-token global block of this layer (the latent amortized over the dies + pool keys)."""
        return BLOCK * KCOL * 2 / self.tp + 32 * self.ID * 2

    def _local_rows(self, phys: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Global physical latent rows [n] -> per-die local rows [tp, n] (0xFFFFFFFF where the die does not own it)."""
        blk, off = phys // BLOCK, phys % BLOCK
        out = []
        for r in range(self.tp):
            own = valid & (blk % self.tp == r)
            out.append(torch.where(own, (blk // self.tp) * BLOCK + off, torch.full_like(phys, 0xFFFFFFFF)))
        return torch.stack(out).to(torch.int64)

    def _local_pt(self, full: torch.Tensor) -> torch.Tensor:
        """[S, MB] completed global block table -> the per-die tables [tp * S, MB + 1] (see the module docstring)."""
        S = full.shape[0]
        full = full.long()
        out = []
        for r in range(self.tp):
            own = (full % self.tp == r) & (full != self.dummy_block)
            lp = torch.where(own, full // self.tp, torch.full_like(full, self.null_block))
            lp[:, self.MB - 1] = self.sink_block
            out.append(torch.cat([lp, torch.full((S, 1), self.null_block, dtype=torch.long)], dim=1))
        return torch.cat(out, dim=0).to(torch.int32)

    # ------------------------------------------------------------------------------------------- caches
    def allocate_decode_state(self, batch: int | None = None, num_blocks: int | None = None):
        """GlmDSA.allocate_decode_state with the per-die latent pool and page tables."""
        if num_blocks is not None and self.latent is None:
            self.NB = int(num_blocks)
            self.dummy_block = self.NB - 1
        if self.latent is None:
            S = self.S
            self.NBL = -(-(self.NB - 1) // self.tp)
            self.null_block, self.sink_block = self.NBL, self.NBL + 1
            self.latent = ttnn.zeros(
                (1, 1, (self.NBL + 2) * BLOCK, KCOL),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
                memory_config=DRAM,
            )
            # NULL block rows (kill column) and the SINK row
            special = torch.zeros(BLOCK + 1, KCOL)
            special[:BLOCK, C_KILL] = -KILL
            special[BLOCK, C_SINK] = 1.0
            src = ttnn.from_torch(
                special, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh, mesh_mapper=self._rep
            )
            rows = torch.cat([self.null_block * BLOCK + torch.arange(BLOCK), torch.tensor([self.sink_block * BLOCK])])
            idx = self._dev(rows.reshape(1, -1).to(torch.int64), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.scatter_rows(self.latent, src, idx)
            ttnn.deallocate(src)
            ttnn.deallocate(idx)
            # replicated pool keys, global page table and open-pool state: as GlmDSA
            self.KB = -(-self.NB * 32 // self.P_alloc)
            self.pooled = ttnn.zeros(
                (self.KB * self.P_alloc // 32, 1, 32, self.ID),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=DRAM,
            )
            fits = self.S * (self.T // BLOCK) + 1 <= self.NB
            pt0 = (
                self.default_page_table()
                if fits
                else self.complete_page_table(torch.zeros(self.S, 0, dtype=torch.int32))
            )
            self.page_table = self._dev(pt0, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            self.page_table_host = pt0
            self.lpt = self._dies(self._local_pt(pt0), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            z = lambda: ttnn.zeros(
                (1, 1, S, self.ID), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=DRAM
            )
            self.acc_m, self.acc_den, self.acc_num = z(), z(), z()
        return self.latent

    def set_page_table(self, pt: torch.Tensor):
        super().set_page_table(pt)
        ttnn.copy_host_to_device_tensor(
            self._dies(self._local_pt(self.page_table_host), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, host=True), self.lpt
        )

    # ----------------------------------------------------------------------------------------- constants
    def _shard_block_rows(self, rows: int) -> int:
        rb = min(rows, max(32, self.shard_rows // 32 * 32))
        while rows % rb:
            rb -= 32
        return rb

    def prepare_constants(self, prefill_lengths):
        super().prepare_constants(prefill_lengths)
        rank = torch.arange(self.tp, dtype=torch.float32).reshape(self.tp, 1, 1, 1)
        self._const("shard_rank", lambda: self._dies(rank, ttnn.float32, ttnn.TILE_LAYOUT))
        e = lambda col, width, dt: self._dev(
            torch.nn.functional.one_hot(torch.tensor(col), width).float().reshape(1, 1, 1, width), dt
        )
        self._const("shard_e_one", lambda: e(C_ONE, KCOL, ttnn.bfloat16))  # cache rows: + the ONE column
        self._const("shard_e_sink", lambda: e(C_SINK - 512, 32, ttnn.bfloat16))  # query control columns
        self._const("shard_e_kill", lambda: e(C_KILL - 512, 32, ttnn.bfloat16))
        self._const("shard_e0", lambda: e(0, 32, ttnn.float32))
        for rows in {self.S} | {self._shard_block_rows(n) for n in prefill_lengths}:
            pre = torch.empty(1, 1, rows, PREFIX)
            pre[..., 0] = (self.MB - 1) * BLOCK  # -> the SINK block (row 0)
            pre[..., 1:] = self.MB * BLOCK + torch.arange(1, PREFIX).float()  # -> the NULL block
            self._const(("shard_prefix", rows), lambda: self._dev(pre))

    # --------------------------------------------------------------------------------------------- inputs
    def _host_mode(self) -> bool:
        return "_dev" in self.__dict__  # GlmDSA._with_builder shadows _dev with a host builder

    def prefill_inputs(self, n: int, start: int, length: int, slot: int, host: bool = False) -> dict:
        inp = super().prefill_inputs(n, start, length, slot, host)
        if host or not self._host_mode():
            ar = torch.arange(n)
            valid = ar < length
            phys = self._phys_rows(torch.full((n,), slot), start + ar, valid)
            if not host:
                ttnn.deallocate(inp["row_idx"])
            inp["row_idx"] = self._dies(
                self._local_rows(phys, valid), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, host=host or self._host_mode()
            )
        return inp

    def _decode_local_rows(self, positions) -> torch.Tensor:
        S = self.S
        active = torch.tensor([p >= 0 for p in positions])
        pos = torch.tensor([max(p, 0) for p in positions], dtype=torch.long)
        return self._local_rows(self._phys_rows(torch.arange(S), pos, active), active)

    def decode_inputs(self, positions):
        bufs = super().decode_inputs(positions)
        ttnn.deallocate(bufs["row_idx"])
        bufs["row_idx"] = self._dies(self._decode_local_rows(positions), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        return bufs

    def update_decode_inputs(self, bufs, positions):
        for k, (t, dt, lay) in self._decode_host_inputs(positions).items():
            if k == "row_idx":
                h = self._dies(self._decode_local_rows(positions), dt, lay, host=True)
            else:
                h = ttnn.from_torch(t, dtype=dt, layout=lay, mesh_mapper=self._rep)
            ttnn.copy_host_to_device_tensor(h, bufs[k])

    def local_rows_device(self, blk, in_blk, act):
        """tt/decode_inputs.DeviceDecodeInputs: per-die latent rows of the decoded tokens from the global block ids
        (fp32 [1,1,S,1], exact below 2**24) inside the decode trace."""
        q = ttnn.floor(ttnn.multiply(blk, 1.0 / self.tp, memory_config=DRAM), memory_config=DRAM)  # tp: power of two
        own = ttnn.subtract(blk, ttnn.multiply(q, float(self.tp)), memory_config=DRAM)
        mine = ttnn.multiply(ttnn.eq(own, self._const("shard_rank")), act, memory_config=DRAM)
        rows = ttnn.add(ttnn.multiply(q, float(BLOCK)), in_blk, memory_config=DRAM)
        out = ttnn.where(mine, rows, SENTINEL_F, memory_config=DRAM)
        for t in (q, own, mine, rows):
            ttnn.deallocate(t)
        return out

    # ------------------------------------------------------------------------------------------ writes
    def _write_latent(self, c, rows: int, row_idx):
        """c [1,1,rows,512] bf16 TILE -> [rows, 544] (ONE column set) -> the owner die's rows (others skip)."""
        cp = ttnn.pad(c, [(0, 0), (0, 0), (0, 0), (0, KCOL - self.R)], 0.0)
        cx = ttnn.add(cp, self._const("shard_e_one"), memory_config=DRAM)
        ttnn.deallocate(cp)
        c_rm = ttnn.to_layout(cx, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(cx)
        ttnn.scatter_rows(self.latent, ttnn.reshape(c_rm, (rows, KCOL)), row_idx)
        ttnn.deallocate(c_rm)

    # --------------------------------------------------------------------------------------- attention
    def _sparse_attend(self, q_abs, index_f32, cache_batch_idx=None, latent_rows=None):
        assert latent_rows is not None, "the sharded attention needs the query rows' own latent (sink logit)"
        rows = int(q_abs.shape[2])
        rb = self._shard_block_rows(rows)
        if rb == rows:
            return self._shard_attend(q_abs, index_f32, cache_batch_idx, latent_rows, rows)
        H, W = int(q_abs.shape[1]), int(index_f32.shape[-1])
        parts = []
        for r0 in range(0, rows, rb):
            q = ttnn.slice(q_abs, (0, 0, r0, 0), (1, H, r0 + rb, self.R), memory_config=DRAM)
            ix = ttnn.slice(index_f32, (0, 0, r0, 0), (1, 1, r0 + rb, W), memory_config=DRAM)
            c = ttnn.slice(latent_rows, (0, 0, r0, 0), (1, 1, r0 + rb, self.R), memory_config=DRAM)
            parts.append(self._shard_attend(q, ix, cache_batch_idx, c, rb))
            for t in (q, ix, c):
                ttnn.deallocate(t)
        out = ttnn.concat(parts, dim=2, memory_config=DRAM)
        for t in parts:
            ttnn.deallocate(t)
        return out

    def _shard_attend(self, q_abs, index_f32, cache_batch_idx, c, rows: int):
        H, tp, R = int(q_abs.shape[1]), self.tp, self.R
        # 1. query control columns [0, q.c_self, 1, 0 ...] and the all-gather of every head
        prod = ttnn.multiply(q_abs, c, memory_config=DRAM)  # c broadcast over the heads
        sd = ttnn.sum(prod, dim=-1, keepdim=True, memory_config=DRAM)  # [1,H,rows,1] raw own-key logit
        ttnn.deallocate(prod)
        ts = ttnn.multiply(sd, self._const("shard_e_sink"), memory_config=DRAM)  # [1,H,rows,32]
        ttnn.deallocate(sd)
        tail = ttnn.add(ts, self._const("shard_e_kill"), memory_config=DRAM)
        ttnn.deallocate(ts)
        qx = ttnn.concat([q_abs, tail], dim=-1, memory_config=DRAM)  # [1,H,rows,544]
        ttnn.deallocate(tail)
        qx2 = ttnn.reshape(qx, (1, 1, H * rows, KCOL))
        qg = self.sccl.all_gather(qx2, dim=1, dtype=ttnn.bfloat16, decode=False)  # [1,tp,H*rows,544]
        ttnn.deallocate(qx)  # (qx2 is a view of it or a temporary)
        q_rm0 = ttnn.to_layout(qg, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        self.sccl.release(qg)
        q_rm = ttnn.reshape(q_rm0, (1, tp * H, rows, KCOL))
        # 2. index list with the [sink, null...] prefix; attention over this die's keys
        ix = ttnn.concat([self._const(("shard_prefix", rows)), index_f32], dim=-1, memory_config=DRAM)
        idx = ttnn.to_layout(ttnn.typecast(ix, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(ix)
        o = ttnn.transformer.sparse_sdpa(
            q_rm,
            self.latent,
            idx,
            KCOL,
            kv_format=ttnn.SparseKVFormat.BF16,
            scale=self.scale,
            k_chunk_size=32,
            compute_kernel_config=self.sdpa_ckc,
            cache_batch_idx=cache_batch_idx,
            page_table=self.lpt,
            page_block_size=BLOCK,
        )  # [1, tp*H, rows, 544] bf16 ROW_MAJOR
        ttnn.deallocate(idx)
        ttnn.deallocate(q_rm0)
        # 3. per-die partials in the common sink unit, reduce-scatter over the head groups: the numerator T = o / w
        # ([1,tp,n,544], columns >= 512 unused) and the softmax mass / w (one 32-column tile, column 0)
        n = H * rows
        ot = ttnn.to_layout(ttnn.reshape(o, (1, tp, n, KCOL)), ttnn.TILE_LAYOUT, memory_config=DRAM)
        ttnn.deallocate(o)
        u = ttnn.typecast(ttnn.slice(ot, (0, 0, 0, C_ONE), (1, tp, n, C_ONE + 1), memory_config=DRAM), ttnn.float32)
        w0 = ttnn.typecast(ttnn.slice(ot, (0, 0, 0, C_SINK), (1, tp, n, C_SINK + 1), memory_config=DRAM), ttnn.float32)
        w = ttnn.maximum(w0, TINY)
        small = ttnn.lt(w, 0.5, memory_config=DRAM)
        one_m_w = ttnn.rsub(w, 1.0, memory_config=DRAM)
        mass = ttnn.where(small, one_m_w, u, memory_config=DRAM)
        rw = ttnn.reciprocal(w, memory_config=DRAM)
        T = ttnn.multiply(ot, rw, memory_config=DRAM, dtype=self.shard_rs_dtype)  # [1,tp,n,544]
        ttnn.deallocate(ot)
        m1 = ttnn.multiply(mass, rw, memory_config=DRAM)  # [1,tp,n,1]
        m32 = ttnn.multiply(m1, self._const("shard_e0"), memory_config=DRAM)  # [1,tp,n,32], column 0
        for t in (u, w0, w, small, one_m_w, mass, rw, m1):
            ttnn.deallocate(t)
        num_all = self._reduce_scatter_heads(T)  # [1,1,n,544]: this die's heads, summed over the dies
        z_all = self._reduce_scatter_heads(m32)  # [1,1,n,32]
        num = ttnn.slice(num_all, (0, 0, 0, 0), (1, 1, n, R), memory_config=DRAM)
        z0 = ttnn.slice(z_all, (0, 0, 0, 0), (1, 1, n, 1), memory_config=DRAM)
        ttnn.deallocate(num_all)
        ttnn.deallocate(z_all)
        z = ttnn.reciprocal(ttnn.maximum(z0, TINY), memory_config=DRAM)
        ttnn.deallocate(z0)
        o_lat = ttnn.multiply(num, z, memory_config=DRAM, dtype=ttnn.bfloat16)
        for t in (num, z):
            ttnn.deallocate(t)
        return ttnn.reshape(o_lat, (1, H, rows, R))

    def _reduce_scatter_heads(self, t):
        ccl = self.sccl
        out = ttnn.experimental.reduce_scatter_minimal_async(
            t,
            persistent_output_buffers=None,
            dim=1,
            multi_device_global_semaphore=ccl.tt_ccl.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=ccl.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=ccl.links() or ccl.tt_ccl.get_num_links(),
            memory_config=DRAM,
            intermediate_memory_config=DRAM,
            topology=ccl.topology(),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        ttnn.deallocate(t)
        return out


_CLASSES = {}


def install_dsa_shard(dsa, ccl):
    """Swap a (Multichip)GlmDSA instance to the block-sharded cache. Call before allocate_decode_state."""
    base = dsa.__class__
    if not issubclass(base, ShardedDSAMixin):
        cls = _CLASSES.get(base)
        if cls is None:
            cls = _CLASSES[base] = type("Sharded" + base.__name__, (ShardedDSAMixin, base), {})
        dsa.__class__ = cls
    dsa.shard_init(ccl)
    return dsa


def dsa_bytes_per_token(dsa_layers) -> float:
    """DRAM per die per token of the DSA caches of the given GlmDSA layers (sharded or replicated)."""
    tot = 0.0
    for a in dsa_layers:
        if getattr(a, "dsa_shard", False):
            tot += a.dsa_bytes_per_block() / BLOCK
        else:
            lat = 1 if str(getattr(a, "latent_dtype", "")).endswith("FP8_E4M3") else 2  # stage-8 fp8 latent pool
            tot += (BLOCK * a.R * lat + 32 * a.ID * 2) / BLOCK
    return tot


assert IDX_WIDTH % 32 == 0
