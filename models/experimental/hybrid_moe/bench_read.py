# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Microbenchmark: the host step's small device -> host reads (die 0 of a replicated row-major tensor), idle device;
to_torch of a [1, 1, rows, width] tensor vs one read of the two packed. usage (in the image): python bench_read.py"""

import time

import torch

import ttnn


def t_read(mesh, t, n=200):
    d0 = ttnn.get_device_tensors(t)[0]
    ttnn.to_torch(d0)
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(n):
        ttnn.to_torch(ttnn.get_device_tensors(t)[0])
    return (time.perf_counter() - t0) / n * 1e6


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=10_000_000)
    rep = ttnn.ReplicateTensorToMesh(mesh)
    for name, shape, dt in (
        ("logits fp32 4x288", (1, 1, 4, 288), ttnn.float32),
        ("x bf16 4x3072", (1, 1, 4, 3072), ttnn.bfloat16),
        ("x bf16 32x3072", (1, 1, 32, 3072), ttnn.bfloat16),
        ("packed bf16 4x3360", (1, 1, 4, 3360), ttnn.bfloat16),
    ):
        t = ttnn.from_torch(
            torch.randn(shape),
            dtype=dt,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=rep,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tl = ttnn.from_torch(
            torch.randn(shape),
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=rep,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        print(f"{name:22s} row-major {t_read(mesh, t):7.1f} us   tile {t_read(mesh, tl):7.1f} us", flush=True)
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
