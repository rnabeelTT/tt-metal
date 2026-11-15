# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Is a device-side expert gather (ttnn.slice along the expert dim + ttnn.concat) an exact copy of bfp4 / bfp8 TILE
DRAM weights? usage: ./run_in_image.sh test_expert_gather.py <out dir>"""
import time

import torch

import ttnn

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576)
try:
    torch.manual_seed(0)
    E, H, N = 16, 3072, 256
    ids = [5, 0, 15, 3, 3, 9, -1, -1]  # -1: a zero expert (LRU / sink slot)
    for dt in (ttnn.bfloat4_b, ttnn.bfloat8_b):
        w = torch.randn(1, E, H, N) * 0.02
        t = ttnn.from_torch(
            w,
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
        )
        ref = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1))  # the quantized values
        t0 = time.perf_counter()
        zero = None
        parts = []
        for e in ids:
            if e < 0:
                if zero is None:
                    zero = ttnn.zeros(
                        [1, 1, H, N // 4],
                        dtype=dt,
                        layout=ttnn.TILE_LAYOUT,
                        device=mesh,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                parts.append(zero)
            else:
                parts.append(ttnn.slice(t, [0, e, 0, 0], [1, e + 1, H, N // 4], memory_config=ttnn.DRAM_MEMORY_CONFIG))
        g = ttnn.concat(parts, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.synchronize_device(mesh)
        dt_ms = 1e3 * (time.perf_counter() - t0)
        got = ttnn.to_torch(g, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1))
        want = torch.stack([ref[0, e] if e >= 0 else torch.zeros(H, N) for e in ids]).unsqueeze(0)
        print(
            f"GATHER {dt}: shape {list(g.shape)} dtype {g.dtype} exact {torch.equal(got, want)} "
            f"max |diff| {float((got - want).abs().max()):.3g} ({dt_ms:.1f} ms)",
            flush=True,
        )
finally:
    ttnn.close_mesh_device(mesh)
