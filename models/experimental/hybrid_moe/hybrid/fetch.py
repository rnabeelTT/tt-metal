# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Fetched experts (the FreeToken "hybrid" policy on the fused hybrid MoE): per layer, D of the C device slots are an
LRU cache. At every decode step a share f of the layer's missed experts is copied host -> device into LRU slots and
computed by the device kernels with the resident ones; the CPU computes the rest at the same time. A fetched expert
stays until evicted, so a miss that recurs over the next steps (consecutive tokens share many experts) is a hit.

Slots per layer: [0, S) static (the plan's top S = C - 1 - D), [S, S + D) LRU, C - 1 the all-zero sink.

Device side, per fused layer and step (inside the trace, after the host step that routed the layer): F_MAX
paged_fill_cache writes of the model-wide staging buffers into the layer's [C, 1, H, W] views of its up_gate / down
weights, at the slots in a device page table [F_MAX, 1] the host step wrote; staging rows with nothing to fetch point
at the sink, which a last pair of writes from all-zero staging resets.

Host side: the device-format (bfp4, TP-sharded, rounded like the port's fused weights) copies of every expert that can
miss, loaded from HYB_DEV_EXPERTS (build_dev_experts.py). Which misses to fetch: the most recently active first
(FreeToken's recency rule; ties to the lower id); how many: f x misses, rounded stochastically and capped by F_MAX.

env: HYB_LRU (D slots per layer, 0 = off), HYB_FETCH_FRAC (f, default 0.4), HYB_FETCH_MAX (F_MAX, default 2),
HYB_DEV_EXPERTS (dir of <L>_<e>_gu.tensorbin / <L>_<e>_dn.tensorbin).
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict

import numpy as np
import torch

import ttnn


class FetchCache:
    def __init__(self):
        self.D = int(os.environ.get("HYB_LRU", "0") or 0)
        self.f = float(os.environ.get("HYB_FETCH_FRAC", "0.4"))
        self.F = int(os.environ.get("HYB_FETCH_MAX", "2"))
        self.dir = os.environ.get("HYB_DEV_EXPERTS", "")
        if self.D and not self.dir:
            raise ValueError("HYB_LRU needs HYB_DEV_EXPERTS (the device-format copies of the experts that can miss)")
        if self.D and not (0 < self.F <= self.D):
            raise ValueError(f"HYB_FETCH_MAX={self.F} must be in 1..HYB_LRU={self.D}")
        self.rng = np.random.default_rng(int(os.environ.get("HYB_FETCH_SEED", "0")))
        self.lru = {}  # layer -> OrderedDict expert -> slot (least recent first)
        self.free = {}  # layer -> free LRU slots
        self.stamp = {}  # layer -> [E] last step each expert was routed (-1: never)
        self.host = {}  # (layer, expert) -> (gu, dn) host tensors in the device format
        self.step = 0
        self.dev = None  # model-wide staging, created on the first layer (before any capture)

    @property
    def on(self):
        return self.D > 0

    # -- setup ------------------------------------------------------------------------------------------------------
    def plan_layer(self, layer, E, C, static_ids):
        """The slot list of a layer: static experts, D empty LRU slots (zero experts), the sink."""
        S = C - 1 - self.D
        self.lru[layer] = OrderedDict()
        self.free[layer] = list(range(S, S + self.D))
        self.stamp[layer] = np.full(E, -1, dtype=np.int64)
        return [int(i) for i in static_ids[:S]] + [-1] * self.D + [-1]

    def load_host(self, layer, experts):
        """The device-format copies of a layer's missable experts (host RAM; read once at build)."""
        for e in experts:
            p = os.path.join(self.dir, f"{layer}_{e}")
            self.host[(layer, e)] = (ttnn.load_tensor(p + "_gu.tensorbin"), ttnn.load_tensor(p + "_dn.tensorbin"))

    def init_layer(self, mod):
        """Per fused layer, before any capture: the [C, 1, *, *] views of its expert weights, and the model-wide
        staging on the first call."""
        ug, dn = mod.up_gate, mod.down
        gs, ds = [int(v) for v in ug.shape], [int(v) for v in dn.shape]
        mod._fetch_views = (ttnn.reshape(ug, [gs[1], 1, gs[2], gs[3]]), ttnn.reshape(dn, [ds[1], 1, ds[2], ds[3]]))
        if self.dev is not None:
            return
        mesh = self.mesh = mod.device
        kw = dict(layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        g_sh, d_sh = ttnn.ShardTensorToMesh(mesh, dim=-1), ttnn.ShardTensorToMesh(mesh, dim=-2)

        def stage(shape, mapper, dt):
            return ttnn.from_torch(torch.zeros(*shape), dtype=dt, mesh_mapper=mapper, **kw)

        tp = len(mesh.get_device_ids())  # shapes above are per device; the staging is built from the full tensor
        gsh, dsh = [1, 1, gs[2], tp * gs[3]], [1, 1, tp * ds[2], ds[3]]
        gdt, ddt = ug.dtype, dn.dtype
        rep = ttnn.ReplicateTensorToMesh(mesh)
        self.dev = {
            "gu": [stage(gsh, g_sh, gdt) for _ in range(self.F)],
            "dn": [stage(dsh, d_sh, ddt) for _ in range(self.F)],
            "zero_gu": stage(gsh, g_sh, gdt),
            "zero_dn": stage(dsh, d_sh, ddt),
            "pt": ttnn.from_torch(
                torch.zeros(self.F, 1, dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                mesh_mapper=rep,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            "pt_sink": ttnn.from_torch(
                torch.full((1, 1), gs[1] - 1, dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                mesh_mapper=rep,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            "C": gs[1],
        }

    # -- device side (traced) -----------------------------------------------------------------------------------------
    def device_writes(self, mod):
        """The staged experts into their slots, then the sink back to zero (traced with the layer's experts)."""
        d = self.dev
        gv, dv = mod._fetch_views
        for j in range(self.F):
            ttnn.experimental.paged_fill_cache(gv, d["gu"][j], d["pt"], batch_idx=j)
            ttnn.experimental.paged_fill_cache(dv, d["dn"][j], d["pt"], batch_idx=j)
        ttnn.experimental.paged_fill_cache(gv, d["zero_gu"], d["pt_sink"], batch_idx=0)
        ttnn.experimental.paged_fill_cache(dv, d["zero_dn"], d["pt_sink"], batch_idx=0)

    # -- host side (between trace segments) ---------------------------------------------------------------------------
    def choose(self, layer, ids, slot_of):
        """After routing a layer's step (ids [m, k]): LRU hits refreshed, the fetch set chosen, slot_of updated in place
        (fetched experts now resident at their LRU slots). Returns (fetched ids, their slots) for stage()."""
        self.step += 1
        lru, free, stamp = self.lru[layer], self.free[layer], self.stamp[layer]
        picked = np.unique(ids)
        for e in picked:
            if e in lru:
                lru.move_to_end(e)
        miss = [int(e) for e in picked if slot_of[e] < 0]
        x = self.f * len(miss)
        nf = min(int(x) + int(self.rng.random() < x - int(x)), self.F, len(miss))
        fetched, slots = [], []
        if nf:
            miss.sort(key=lambda e: (-stamp[e], e))  # most recently active first
            protect = set(int(e) for e in picked)
            for e in miss[:nf]:
                if free:
                    s = free.pop()
                else:
                    victim = next((v for v in lru if v not in protect), None)
                    if victim is None:
                        break
                    s = lru.pop(victim)
                    slot_of[victim] = -1
                lru[e] = s
                slot_of[e] = s
                fetched.append(e)
                slots.append(s)
        stamp[picked] = self.step
        return fetched, slots

    def stage(self, layer, fetched, slots, st):
        """The fetched experts into the staging buffers and their slots into the page table (host -> device copies on
        CQ0, ahead of the segment that runs device_writes)."""
        t0 = time.perf_counter()
        d = self.dev
        for j, (e, s) in enumerate(zip(fetched, slots)):
            gu, dn = self.host[(layer, e)]
            ttnn.copy_host_to_device_tensor(gu, d["gu"][j])
            ttnn.copy_host_to_device_tensor(dn, d["dn"][j])
        pt = torch.full((self.F, 1), d["C"] - 1, dtype=torch.int32)  # unused rows: the sink (reset right after)
        pt[: len(slots), 0] = torch.tensor(slots, dtype=torch.int32)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)
            ),
            d["pt"],
        )
        st["t_fetch"] += time.perf_counter() - t0
        st["fetched"] += len(fetched)
