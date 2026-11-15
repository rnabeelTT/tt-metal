# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Microbench of the M2.7 LM head as DFlash runs it (vocab-sharded over the 4 dies: [32 rows, 3072] x [3072, 50016]
bfp8 per die, ttnn.linear into DRAM) and of the sampled path's candidates (dflash_tt.cands_body: 2 x 32768-wide top-64).
Measured 2026-11-14 (runs/m27/lmbench): LM head 0.47 ms per call (163 MB per die, ~350 GB/s; gpt-oss's streamed LM
head reaches ~508 GB/s: < 0.2 ms to gain per call); candidates 2 chunks x top-64 0.66 ms, 4 x 64 0.91, 8 x 64 1.30,
4 x 32 0.60, 8 x 32 0.80, 16 x 32 1.11 (exactness for top-k 40 needs k = 64: 2 x 64 is the choice).
usage: FORK=1 ./run_in_image.sh bench_lm_head.py <out dir>"""
import time

import torch

import ttnn
from dflash_tt import cands_body

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576)
try:
    H, V = 3072, 200064
    w = ttnn.from_torch(
        torch.randn(H, V) * 0.02,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
    )
    x = ttnn.from_torch(
        torch.randn(1, 1, 32, H),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    def timed(name, fn, n=20):
        outs = fn()
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(n):
            outs = fn()
        ttnn.synchronize_device(mesh)
        dt = (time.perf_counter() - t0) / n
        print(f"LMBENCH {name}: {1e3 * dt:.3f} ms per call", flush=True)
        return outs

    lg = timed("lm_head linear (eager)", lambda: ttnn.linear(x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG))
    timed("candidates top-64 x 2 chunks (eager)", lambda: cands_body(lg))
    # traced, as served
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    lg2 = ttnn.linear(x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    timed("lm_head linear (trace)", lambda: ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False))
    tid2 = ttnn.begin_trace_capture(mesh, cq_id=0)
    c = cands_body(lg2)
    ttnn.end_trace_capture(mesh, tid2, cq_id=0)
    timed("candidates (trace)", lambda: ttnn.execute_trace(mesh, tid2, cq_id=0, blocking=False))
    import dflash_tt

    for C, K in ((2, 64), (4, 64), (8, 64), (4, 32), (8, 32), (16, 32)):
        dflash_tt.CAND_CHUNKS, dflash_tt.CAND_K = C, K
        timed(f"candidates chunks {C} k {K} (eager)", lambda: cands_body(lg))
    print(f"LMBENCH weight bytes per die {H * V // 4 * 1088 // 1024 / 1e6:.0f} MB", flush=True)
finally:
    ttnn.close_mesh_device(mesh)
