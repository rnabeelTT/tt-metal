# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefix-cache snapshots of the per-sequence state that vLLM's KV blocks do not hold (the plugin's prefix_state slots).

vLLM pages the global layers' KV (tt/attention.GlobalPages) and its prefix cache shares those blocks between requests.
Resuming a prompt at a cached position q also needs, for every one of the 48 applications:

  * the four short-conv states at q (k, v, attention output, MoE output: the last 3 inputs, fp32 row-major), and
  * the sliding layers' last window of K / V (positions [q - W, q): W / 64 blocks of the slot's ring).

A snapshot slot holds both in device pools allocated at start-up (before any trace capture); saving and restoring are
one block-copy program each (tt/blkcopy.py). The conv states at q come from the prefill chunk that holds q: the convs
slice them out of their inputs (ShortConv / DynShortConv snap_row). The ring is copied after that chunk (a ring holds
the window plus a chunk plus a block, so [q - W, q) is still in it).

Device memory per snapshot slot: 40 sliding applications x 2 x W / 64 blocks (bf16 KV: 168 MB on one die, divided by
the tensor-parallel degree) + the conv states (~1 MB)."""

from __future__ import annotations

import torch

import ttnn

from .blkcopy import BlockCopy

DRAM = ttnn.DRAM_MEMORY_CONFIG


def _tiles_per_block(t) -> int:
    s = t.padded_shape
    return int(s[1]) * int(s[2]) * int(s[3]) // 1024


class PrefixStore:
    def __init__(self, model, n_slots: int):
        self.m, self.n = model, int(n_slots)
        mesh = model.mesh
        N = mesh.get_num_devices()
        self.convs = []  # per application: its 4 conv objects (lane states)
        self.cpool = []  # per application: 4 pools [1, 1, 3 * n, C] fp32 row-major (sharded like the states)
        self.rings = []  # (cache tensor, pool tensor, AttnCache) per sliding application and k / v
        for a, st in enumerate(model.apps):
            c = st.cache
            convs = list(st.lanes[0])
            self.convs.append([list(ln) for ln in st.lanes])
            pools = []
            for i, cv in enumerate(convs):  # k / v states sharded over the kv heads, the hidden-size ones replicated
                C = int(cv.st.padded_shape[-1])
                shard = i < 2 and N > 1
                mapper = (ttnn.ShardTensorToMesh(mesh, dim=3) if shard else ttnn.ReplicateTensorToMesh(mesh)) \
                    if N > 1 else None
                pools.append(ttnn.from_torch(torch.zeros(1, 1, 3 * self.n, C * (N if shard else 1)),
                                             dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                                             memory_config=DRAM, mesh_mapper=mapper))
            self.cpool.append(pools)
            if c.w.sliding:
                self.WB = c.w.W // c.block
                for t in (c.k_cache, c.v_cache):
                    s = t.padded_shape
                    z = torch.zeros(self.n * self.WB, int(s[1]) * N, int(s[2]), int(s[3]))
                    pool = ttnn.from_torch(z, dtype=t.dtype, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM,
                                           mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1) if N > 1 else None)
                    self.rings.append((t, pool, c))
        self.bc = BlockCopy(mesh, self.rings[0][0])
        self.pending = None

    # -- save ----------------------------------------------------------------------------------------------------------
    def arm(self, row: int, lane: int = 0):
        """The next prefill chunk holds the snapshot position at row `row` (0 .. valid rows) of lane `lane`'s
        segment."""
        for lanes in self.convs:
            for cv in lanes[lane]:
                cv.snap_row, cv.snap_out = row, None

    def save(self, k: int, slot: int, q: int, lane: int = 0):
        """After the armed chunk: snapshot slot k <- the conv states it sliced and slot `slot`'s ring window before q."""
        jobs = []
        for lanes, pools in zip(self.convs, self.cpool):
            for cv, pool in zip(lanes[lane], pools):
                assert cv.snap_out is not None, "a conv the armed chunk did not run"
                jobs.append((cv.snap_out, 0, pool, 3 * k, 3))
        jobs += self._ring_jobs(slot, q, k, save=True)
        self.bc(jobs)
        for lanes in self.convs:
            for cv in lanes[lane]:
                cv.snap_row, cv.snap_out = None, None

    def _ring_jobs(self, slot, q, k, save):
        jobs = []
        jq = q // 64
        lo = max(0, jq - self.WB)
        for t, pool, c in self.rings:
            tpb = _tiles_per_block(t)
            per = c.per_slot
            for j in range(lo, jq):
                ring_blk = slot * per + j % per
                pool_blk = k * self.WB + (j - (jq - self.WB))
                if save:
                    jobs.append((t, ring_blk * tpb, pool, pool_blk * tpb, tpb))
                else:
                    jobs.append((pool, pool_blk * tpb, t, ring_blk * tpb, tpb))
        return _merge(jobs)

    # -- restore -------------------------------------------------------------------------------------------------------
    def restore(self, k: int, slot: int, q: int, lane: int = 0):
        """Snapshot slot k (taken at q) -> lane `lane`'s conv states and slot `slot`'s ring (after the lane's reset)."""
        jobs = []
        for lanes, pools in zip(self.convs, self.cpool):
            for cv, pool in zip(lanes[lane], pools):
                jobs.append((pool, 3 * k, cv.st, 0, 3))
        jobs += self._ring_jobs(slot, q, k, save=False)
        self.bc(jobs)

    def warm(self):
        """Compile the block-copy program (before any trace capture): a no-op copy of snapshot 0 onto itself."""
        t, pool, _ = self.rings[0]
        self.bc([(pool, 0, pool, 0, 1)])


def _merge(jobs):
    """Adjacent page runs of the same tensor pair -> one job."""
    out = []
    for j in jobs:
        if out:
            s, sp, d, dp, n = out[-1]
            if s is j[0] and d is j[2] and sp + n == j[1] and dp + n == j[3]:
                out[-1] = (s, sp, d, dp, n + j[4])
                continue
        out.append(j)
    return out
