# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-step decode inputs of the full model, derived on device from one position tensor (trace-safe, no host work per
token).

Every DSA layer reads the same per-step inputs (cache row of the new token, open-pool bookkeeping, causal limits, the
dense-phase index list), and every KDA layer reads the same slot activity masks. The model keeps ONE set of fixed-address
buffers per kind, shared by all layers. Inside the decode trace, `refresh()` recomputes the DSA fields from the device
position tensor `pos` [1,1,S,1] fp32 (-1 = inactive slot) and the device copy of the page table, and `advance()` moves
every active slot one position forward after the step. The host writes `pos` and the page table only when slots
start, stop or get new blocks (outside the trace).

The fields match tt/dsa.GlmDSA._decode_host_inputs exactly (tests/test_model_probe.py::test_decode_inputs_match_host
compares them; the block-sharded layout's per-die row_idx against ShardedDSAMixin._decode_local_rows on every die):
  row_idx    physical latent row page_table[s, p // 128] * 128 + p % 128 (uint32 [1,S]; 0xFFFFFFFF = inactive)
A slot whose position reaches max_context (T) is treated as inactive (decode-overrun guard: column T/128 of the page
table is the shared dummy block and later columns resolve to block 0, another slot's cache; the generator refuses such
positions on the host, and a device-advanced position past T never writes a cache row).
  pool_pos   p // 4 (int32 [S]; -1 = inactive)
  active, reset (p % 4 == 0), ape_sel (ape[p % 4]), pos, limit ((p + 1) // 4, 0 inactive) + its uint32 copy,
  dense (p <= DENSE_MAX_POSITION), dense_idx (col if col <= p else sentinel).
Positions and block ids stay below 2**24, so fp32 holds them exactly.
"""

from __future__ import annotations

import os

import torch

import ttnn

from .dsa import BLOCK, DENSE_MAX_POSITION, IDX_WIDTH, KPOOL, SENTINEL_F

DRAM = ttnn.DRAM_MEMORY_CONFIG


class DeviceDecodeInputs:
    def __init__(self, mesh, dsa=None, kda=None, batch: int = 32):
        """dsa: any GlmDSA layer of the model (shared geometry, ape table); kda: any GlmKDA layer (None if absent)."""
        self.mesh = mesh
        self.S = batch
        self.dsa, self.kda = dsa, kda
        self._rep = ttnn.ReplicateTensorToMesh(mesh)
        self.positions = [-1] * batch
        self.pos = self._dev(torch.full((1, 1, batch, 1), -1.0))
        self.kda_bufs = kda.decode_inputs(self.positions) if kda is not None else None
        self.dsa_bufs = None
        if dsa is not None:
            self.dsa_bufs = dsa.decode_inputs(self.positions)
            self.MB = dsa.MB
            self.pt = self._dev(dsa.page_table_host.float().reshape(1, 1, batch, self.MB))
            self.col_mb = self._dev(torch.arange(self.MB).float().reshape(1, 1, 1, self.MB))
            self.col_idx = self._dev(torch.arange(IDX_WIDTH).float().reshape(1, 1, 1, IDX_WIDTH))
            self.ape = [self._dev(dsa.ape[j].float().reshape(1, 1, 1, -1)) for j in range(KPOOL)]

    def _dev(self, t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=self.mesh, mesh_mapper=self._rep)

    def _host(self, t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, mesh_mapper=self._rep)

    # ------------------------------------------------------------------ host side (outside traces)
    def set_positions(self, positions):
        """Positions of the token each slot decodes next (-1 = inactive). Slot start / stop only."""
        assert len(positions) == self.S
        self.positions = list(positions)
        ttnn.copy_host_to_device_tensor(
            self._host(torch.tensor(positions, dtype=torch.float32).reshape(1, 1, self.S, 1)), self.pos
        )
        if self.kda is not None:
            self.kda.update_decode_inputs(self.kda_bufs, self.positions)
        if self.dsa is not None:  # eager fill (a trace replay refreshes them itself)
            self.dsa.update_decode_inputs(self.dsa_bufs, self.positions)

    def set_page_table(self, page_table_full: torch.Tensor):
        """[S, MB] completed block table (GlmDSA.complete_page_table), copied after every DSA layer installed it."""
        if self.dsa is not None:
            ttnn.copy_host_to_device_tensor(self._host(page_table_full.float().reshape(1, 1, self.S, self.MB)), self.pt)

    def host_advance(self, n: int = 1):
        """Host mirror of `advance()` (the generator tracks positions for its own bookkeeping)."""
        self.positions = [p + n if p >= 0 else p for p in self.positions]

    # ------------------------------------------------------------------ device side (inside the decode trace)
    def _into(self, src, key, dtype, layout, shape=None):
        """src -> the fixed buffer dsa_bufs[key] (dtype / layout / shape converted). Only the typecast / to_layout
        outputs are freed here: a reshape is a VIEW of its input, and ttnn.deallocate (force=True) of a view frees the
        storage of `src`, which the caller still owns (and frees itself)."""
        dst = self.dsa_bufs[key]
        owned = []
        t = src
        if shape is not None:
            t = ttnn.reshape(t, shape)
        if t.dtype != dtype:
            t = ttnn.typecast(t, dtype)
            owned.append(t)
        if t.layout != layout:
            t = ttnn.to_layout(t, layout)
            owned.append(t)
        if tuple(t.shape) != tuple(dst.shape):
            t = ttnn.reshape(t, tuple(dst.shape))
        ttnn.copy(t, dst)
        del t  # drop a reshape view of an owned tensor before freeing it (its storage is the owned tensor's)
        for o in owned:
            ttnn.deallocate(o)

    def refresh(self):
        """Recompute the DSA per-step inputs from `pos` and the page table (in place)."""
        if self.dsa is None:
            return
        S = self.S
        if os.environ.get("GLM53_DECODE_GUARD", "1") == "1":
            act = ttnn.multiply(
                ttnn.ge(self.pos, 0.0, memory_config=DRAM), ttnn.lt(self.pos, float(self.dsa.T)), memory_config=DRAM
            )  # 1.0 / 0.0: 0 <= pos < max_context
        else:
            act = ttnn.ge(self.pos, 0.0, memory_config=DRAM)
        p = ttnn.relu(self.pos, memory_config=DRAM)
        pool = ttnn.floor(ttnn.multiply(p, 1.0 / KPOOL, memory_config=DRAM), memory_config=DRAM)  # p // 4
        j = ttnn.subtract(p, ttnn.multiply(pool, float(KPOOL)), memory_config=DRAM)  # p % 4
        blk_col = ttnn.floor(ttnn.multiply(p, 1.0 / BLOCK), memory_config=DRAM)  # p // 128
        # page_table[s, p // 128]: one-hot over the block columns (elementwise; exact in fp32)
        onehot = ttnn.eq(self.col_mb, blk_col, memory_config=DRAM)  # [1,1,S,MB]
        blk = ttnn.sum(ttnn.multiply(onehot, self.pt, memory_config=DRAM), dim=-1, keepdim=True)
        ttnn.deallocate(onehot)
        in_blk = ttnn.subtract(p, ttnn.multiply(blk_col, float(BLOCK)), memory_config=DRAM)
        if getattr(self.dsa, "dsa_shard", False):  # block-sharded latent (tt/dsa_shard.py): the owner die's local row
            rows = self.dsa.local_rows_device(blk, in_blk, act)
        else:
            rows = ttnn.add(ttnn.multiply(blk, float(BLOCK)), in_blk, memory_config=DRAM)
            rows = ttnn.where(act, rows, SENTINEL_F, memory_config=DRAM)  # 2**32 saturates to 0xFFFFFFFF
        self._into(rows, "row_idx", ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, shape=(1, 1, 1, S))
        pool_pos = ttnn.where(act, pool, -1.0, memory_config=DRAM)
        self._into(pool_pos, "pool_pos", ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, shape=(1, 1, 1, S))
        self._into(act, "active", ttnn.float32, ttnn.TILE_LAYOUT)
        reset = ttnn.eq(j, 0.0, memory_config=DRAM)
        self._into(reset, "reset", ttnn.float32, ttnn.TILE_LAYOUT)
        ape = None
        for k in range(KPOOL):
            term = ttnn.multiply(ttnn.eq(j, float(k)), self.ape[k], memory_config=DRAM)  # [1,1,S,ID]
            ape = term if ape is None else ttnn.add(ape, term, memory_config=DRAM)
        self._into(ape, "ape_sel", ttnn.float32, ttnn.TILE_LAYOUT)
        self._into(p, "pos", ttnn.float32, ttnn.TILE_LAYOUT)
        lim = ttnn.multiply(ttnn.floor(ttnn.multiply(ttnn.add(p, 1.0), 1.0 / KPOOL)), act, memory_config=DRAM)
        self._into(lim, "limit", ttnn.float32, ttnn.TILE_LAYOUT)
        self._into(lim, "limit_u32", ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, shape=(1, 1, 1, S))
        dense = ttnn.le(p, float(DENSE_MAX_POSITION), memory_config=DRAM)
        self._into(dense, "dense", ttnn.float32, ttnn.TILE_LAYOUT)
        didx = ttnn.where(ttnn.le(self.col_idx, p), self.col_idx, SENTINEL_F, memory_config=DRAM)  # [1,1,S,IDX]
        self._into(didx, "dense_idx", ttnn.float32, ttnn.TILE_LAYOUT)
        for t in (act, p, pool, j, blk_col, blk, in_blk, rows, pool_pos, reset, ape, lim, dense, didx):
            ttnn.deallocate(t)

    def advance(self):
        """pos += 1 for active slots (end of a decode step, inside the trace)."""
        nxt = ttnn.add(self.pos, ttnn.ge(self.pos, 0.0), memory_config=DRAM)
        ttnn.copy(nxt, self.pos)
        ttnn.deallocate(nxt)
