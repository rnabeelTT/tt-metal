# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Microbench of the elastic KV pool resize primitives (bfp8 paged K / V, replicated): ttnn.zeros vs a bf16 device fill
+ typecast, and shrinking by slicing the old pool. usage: ./run_in_image.sh bench_kv_alloc.py <out dir>"""
import time

import torch

import ttnn

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576)
try:
    shape = (1056, 2, 64, 128)  # ~67K tokens of one layer's K at 64-token blocks
    kw = dict(layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def timed(name, fn, n=6):
        fn()  # compile
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        outs = [fn() for _ in range(n)]
        ttnn.synchronize_device(mesh)
        dt = (time.perf_counter() - t0) / n
        for o in outs:
            ttnn.deallocate(o)
        print(f"KVBENCH {name}: {1e3 * dt:.1f} ms per tensor", flush=True)
        return dt

    timed("zeros bfp8", lambda: ttnn.zeros(shape, dtype=ttnn.bfloat8_b, **kw))

    def fill_cast():
        z = ttnn.zeros(shape, dtype=ttnn.bfloat16, **kw)
        c = ttnn.typecast(z, ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(z)
        return c

    timed("zeros bf16 + typecast", fill_cast)
    ref = ttnn.zeros(shape, dtype=ttnn.bfloat8_b, **kw)
    got = fill_cast()
    print(
        f"KVBENCH typecast zeros exact: {torch.equal(ttnn.to_torch(ttnn.get_device_tensors(ref)[0]), ttnn.to_torch(ttnn.get_device_tensors(got)[0]))}"
    )
    big = ttnn.zeros((2048,) + shape[1:], dtype=ttnn.bfloat8_b, **kw)
    timed(
        "slice 2048 -> 1056 blocks",
        lambda: ttnn.slice(big, [0, 0, 0, 0], list(shape), memory_config=ttnn.DRAM_MEMORY_CONFIG),
    )
    small = ttnn.zeros((512,) + shape[1:], dtype=ttnn.bfloat8_b, **kw)
    timed(
        "concat 512 + 544 zeros",
        lambda: ttnn.concat(
            [small, ttnn.zeros((544,) + shape[1:], dtype=ttnn.bfloat8_b, **kw)],
            dim=0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
    )
finally:
    ttnn.close_mesh_device(mesh)
