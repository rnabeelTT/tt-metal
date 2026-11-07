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
        self.host = {}  # (layer, expert) -> (gu, dn) host tensors in the device format (until finalize)
        self.step = 0
        self.dev = None  # model-wide staging, created on the first layer (before any capture)
        # HYB_PINX: the device-format experts as raw device bytes in host memory pinned for the mesh (hybrid/pin):
        # fetch / stream writes read them straight over PCIe (no CPU memcpy into the command-queue hugepage)
        self.pinx_on = bool(self.dir) and os.environ.get("HYB_PINX", "1") == "1"
        self.px = None
        self.raw = {}  # layer -> (region id, {expert: index}, gate|up bytes per die, down bytes per die)
        self.shapes = None  # per-device ([1, 1, H, 2 Nl], [1, 1, Nl, H]) and dtypes of one expert (note_shapes)

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

    def note_shapes(self, mod):
        if self.shapes is None:
            ug, dn = mod.up_gate, mod.down
            gs, ds = [int(v) for v in ug.shape], [int(v) for v in dn.shape]
            self.shapes = ([1, 1, gs[2], gs[3]], [1, 1, ds[2], ds[3]], ug.dtype, dn.dtype, mod.device)

    def finalize(self, loader=None, stream=None):
        """After the model build, before any trace capture (HYB_PINX): every loaded device-format expert -> raw device
        bytes in a pinned per-layer host region (written to a staging tensor the regular way, read back raw), the host
        tensors dropped; destination views for the fetch / stream staging."""
        if not self.pinx_on or self.shapes is None:
            return
        if loader is not None:
            loader.submit(lambda: None).result()  # the queued load_host calls are done
        from .pin import Pinx

        gsh, dsh, gdt, ddt, mesh = self.shapes
        self.mesh = mesh
        px = self.px = Pinx(mesh)
        tp = px.tp
        kw = dict(layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        conv_gu = ttnn.from_torch(
            torch.zeros(gsh[0], gsh[1], gsh[2], tp * gsh[3]),
            dtype=gdt,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
            **kw,
        )
        conv_dn = ttnn.from_torch(
            torch.zeros(dsh[0], dsh[1], tp * dsh[2], dsh[3]),
            dtype=ddt,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-2),
            **kw,
        )
        (dg, ng), (dd, nd) = px.dst(conv_gu), px.dst(conv_dn)
        t0 = time.perf_counter()
        layers = sorted({L for L, _ in self.host})
        for L in layers:
            experts = sorted(e for (l2, e) in self.host if l2 == L)
            rid, _ = px.region(len(experts) * tp * (ng + nd))
            for i, e in enumerate(experts):
                gu, dn = self.host.pop((L, e))
                base = i * tp * (ng + nd)
                ttnn.copy_host_to_device_tensor(gu, conv_gu)
                px.read(dg, rid, [base + d * ng for d in range(tp)])
                ttnn.copy_host_to_device_tensor(dn, conv_dn)
                px.read(dd, rid, [base + tp * ng + d * nd for d in range(tp)])
            self.raw[L] = (rid, {e: i for i, e in enumerate(experts)}, ng, nd)
        ttnn.deallocate(conv_gu)
        ttnn.deallocate(conv_dn)
        if self.dev is not None:
            self.dev["pdst"] = [(px.dst(g)[0], px.dst(d)[0]) for g, d in zip(self.dev["gu"], self.dev["dn"])]
        if stream is not None and stream.dev is not None:
            sg, sd = stream.dev["stage"]
            stream.dev["pdst"] = (px.dst(sg)[0], px.dst(sd)[0])
        print(
            f"HYB_PINX: {sum(len(v[1]) for v in self.raw.values())} experts of {len(layers)} layers pinned "
            f"({time.perf_counter() - t0:.1f} s)",
            flush=True,
        )

    def has_expert(self, layer, e):
        return (layer, e) in self.host or (layer in self.raw and e in self.raw[layer][1])

    def raw_offsets(self, layer, e):
        """(region, gate|up offsets per die, down offsets per die) of a pinned expert."""
        rid, idx, ng, nd = self.raw[layer]
        tp = self.px.tp
        base = idx[e] * tp * (ng + nd)
        return rid, [base + d * ng for d in range(tp)], [base + tp * ng + d * nd for d in range(tp)]

    def write_expert(self, layer, e, dst_pair, host_pair):
        """One expert into a staging pair: zero-copy from the pinned region, else the regular host -> device copy."""
        if self.px is not None and layer in self.raw:
            rid, og, od = self.raw_offsets(layer, e)
            self.px.write(dst_pair[0], rid, og)
            self.px.write(dst_pair[1], rid, od)
            return
        gu, dn = self.host[(layer, e)]
        ttnn.copy_host_to_device_tensor(gu, host_pair[0])
        ttnn.copy_host_to_device_tensor(dn, host_pair[1])

    def release(self):
        if self.px is not None:
            self.px.release()
            self.px = None

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
            self.write_expert(layer, e, d["pdst"][j] if "pdst" in d else None, (d["gu"][j], d["dn"][j]))
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
