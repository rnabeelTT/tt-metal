# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The device drafter's layer program alone (no target model): traced wall of one layer, and its pieces (attention conv,
qkv + norms + rope, attention, o-proj, MLP conv, MLP) with a sync around each (eager). Drafter weights only.

    python -m models.autoports.qwen_qwen3_5_122b_a10b.tools.dflash_drafter_prof
"""
from __future__ import annotations

import time
import types

import torch

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tools.dflash_acceptance import drafter_dir
from models.autoports.qwen_qwen3_5_122b_a10b.tt.dflash_device import ROWS, DFlashDevice


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=128 << 20)
    stub = types.SimpleNamespace(mesh_device=mesh)
    dr = DFlashDevice(stub, drafter_dir())
    rep = ttnn.ReplicateTensorToMesh(mesh)
    x = ttnn.from_torch(torch.randn(1, 1, ROWS, dr.H) * 0.3, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep)
    dr.reset()
    vis = torch.zeros(1, 1, 1, dr.R + ROWS)
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(vis, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), dr.d_vis)
    dr._make_mask()
    lw = dr.layers[0]
    sync = lambda: ttnn.synchronize_device(mesh)
    for _ in range(2):
        ttnn.deallocate(dr.layer_body(lw, ttnn.clone(x)))
    sync()
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    ttnn.deallocate(dr.layer_body(lw, ttnn.clone(x)))
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    sync()
    t0 = time.perf_counter()
    for _ in range(20):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    sync()
    print(f"DRAFTER_PROF {dr.conv_expand} layer traced {(time.perf_counter() - t0) / 20 * 1e3:.3f} ms", flush=True)
    ttnn.release_trace(mesh, tid)
    h = ttnn.clone(x)

    def dyn():
        for t in dr._dyn(lw, "attention_conv", h).values():
            ttnn.deallocate(t)

    def conv_apply():
        kern = dr._dyn(lw, "attention_conv", h)
        ttnn.deallocate(dr._conv_apply(kern, 0, h))
        for t in kern.values():
            ttnn.deallocate(t)

    def mm_gu():
        ttnn.deallocate(dr._mm(h, lw["wgu"]))

    for name, fn in (("conv dyn (4 kernels)", dyn), ("conv dyn + apply", conv_apply), ("mm_gate_up", mm_gu)):
        fn()
        sync()
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        fn()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        sync()
        t0 = time.perf_counter()
        for _ in range(20):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        sync()
        print(f"DRAFTER_PROF {dr.conv_expand} {name} {(time.perf_counter() - t0) / 20 * 1e6:.0f} us", flush=True)
        ttnn.release_trace(mesh, tid)
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
