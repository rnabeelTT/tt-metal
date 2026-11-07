# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Prefill streaming of missed experts (FreeToken's prefill idea, scaled to what a chunk actually routes to): in an
eager prefill MoE, every missed expert with at least HYB_STREAM_MIN tokens is copied host -> device into a model-wide
staging weight tensor ([1, NS, H, W] bfp4, the port's fused layout) and computed there by a second routed-expert call
over the staging; the CPU keeps only the experts with a few tokens. A missed expert costs one ~8 MB copy (~0.4 ms)
instead of a CPU GEMM over its tokens, which for long prompts is the difference between seconds and minutes.

The device-format copies come from HYB_DEV_EXPERTS (build_dev_experts.py; hybrid/fetch.py loads them).
env: HYB_STREAM (1 = on when HYB_DEV_EXPERTS is set), HYB_STREAM_MIN (tokens, default 4), HYB_STREAM_SLOTS (NS, 128).
"""

from __future__ import annotations

import os
import time

import torch

import ttnn


class PrefillStream:
    def __init__(self, fetch):
        self.fetch = fetch  # the device-format host copies (FetchCache.host)
        self.on = bool(fetch.dir) and os.environ.get("HYB_STREAM", "1") == "1"
        self.min_tokens = int(os.environ.get("HYB_STREAM_MIN", "4"))
        self.NS = int(os.environ.get("HYB_STREAM_SLOTS", "128"))
        self.dev = None

    def init(self, mod):
        """Model-wide staging, on the first fused layer (before any capture); per layer, its routed-call attributes."""
        if self.dev is None:
            mesh = mod.device
            tp = len(mesh.get_device_ids())
            gs, ds = [int(v) for v in mod.up_gate.shape], [int(v) for v in mod.down.shape]  # per device
            kw = dict(layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            def zeros(shape, dim, dt):
                return ttnn.from_torch(
                    torch.zeros(*shape), dtype=dt, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim), **kw
                )

            gu = zeros([1, self.NS, gs[2], tp * gs[3]], -1, mod.up_gate.dtype)
            dn = zeros([1, self.NS, tp * ds[2], ds[3]], -2, mod.down.dtype)
            rep = ttnn.ReplicateTensorToMesh(mesh)
            self.dev = {
                "gu": gu,
                "dn": dn,
                "views": (ttnn.reshape(gu, [self.NS, 1, gs[2], gs[3]]), ttnn.reshape(dn, [self.NS, 1, ds[2], ds[3]])),
                "stage": (
                    zeros([1, 1, gs[2], tp * gs[3]], -1, mod.up_gate.dtype),
                    zeros([1, 1, tp * ds[2], ds[3]], -2, mod.down.dtype),
                ),
                "pt": ttnn.from_torch(
                    torch.zeros(1, 1, dtype=torch.int32),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=mesh,
                    mesh_mapper=rep,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
                "ones": ttnn.from_torch(
                    torch.ones(1, 1, 1, self.NS, dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=mesh,
                    mesh_mapper=rep,
                ),
                "mesh": mesh,
            }
        d = self.dev
        mod._stream_attrs = {
            "up_gate": d["gu"],
            "down": d["dn"],
            "ones_mask": d["ones"],
            "num_experts": self.NS,
            "_pairs_masks": {},
            "_dispatch": None,
        }

    def plan(self, layer, idh, miss):
        """Prefill picks idh [rows, k] (torch long) and their miss mask -> (stream mask [rows, k], staging index per
        pick (the other picks: NS - 1, a slot kept all-zero), the experts in staging order): the missed experts with
        >= min_tokens tokens that have host copies, up to NS - 1."""
        ids = idh[miss]
        if ids.numel() == 0:
            return None
        e, n = torch.unique(ids, return_counts=True)
        keep = [
            (int(c), int(x))
            for x, c in zip(e.tolist(), n.tolist())
            if c >= self.min_tokens and self.fetch.has_expert(layer, x)
        ]
        if not keep:
            return None
        keep.sort(reverse=True)  # the most tokens first, if more than NS
        experts = [x for _, x in keep[: self.NS - 1]]  # slot NS - 1 stays empty: the other picks' dummy index
        pos = torch.full((int(idh.max()) + 1,), -1, dtype=torch.long)
        pos[torch.tensor(experts)] = torch.arange(len(experts))
        sidx = pos[idh.clamp(0, pos.numel() - 1)]
        smask = miss & (sidx >= 0)
        return smask, torch.where(smask, sidx, torch.full_like(sidx, self.NS - 1)), experts

    def upload(self, layer, experts, st):
        """The planned experts into staging slots 0.. (eager: host -> device copy, then a device copy into the slot)."""
        t0 = time.perf_counter()
        d = self.dev
        gv, dv = d["views"]
        sg, sd = d["stage"]
        rep = ttnn.ReplicateTensorToMesh(d["mesh"])
        for j, e in enumerate(experts):
            self.fetch.write_expert(layer, e, d.get("pdst"), (sg, sd))
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    torch.tensor([[j]], dtype=torch.int32),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=rep,
                ),
                d["pt"],
            )
            ttnn.experimental.paged_fill_cache(gv, sg, d["pt"], batch_idx=0)
            ttnn.experimental.paged_fill_cache(dv, sd, d["pt"], batch_idx=0)
        st["t_stream"] += time.perf_counter() - t0
        st["streamed"] += len(experts)
