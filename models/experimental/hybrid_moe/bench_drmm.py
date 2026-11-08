# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Microbenchmark: the DFlash drafter's per-die projection matmuls (TP over 4 dies, bfp8 weights, M <= 32 rows),
default ttnn.linear vs explicit 1D (mcast_in0) program configs on n cores vs DRAM-sharded; traced device time per
matmul. usage (in the image): python bench_drmm.py"""

import sys
import time

sys.path.insert(0, "/opt/tt-metal")
import torch  # noqa: E402

import ttnn  # noqa: E402
from models.demos.blackhole.qwen36.tt import tp_common as tpc  # noqa: E402

REPS, RUNS = 20, 5


def timed(mesh, fn):
    fn()  # compile
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(REPS):
        fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(RUNS):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    dt = (time.perf_counter() - t0) / (REPS * RUNS) * 1e6
    ttnn.release_trace(mesh, tid)
    return dt


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=50_000_000)
    grid = mesh.compute_with_storage_grid_size()
    ckc = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True
    )
    rep = ttnn.ReplicateTensorToMesh(mesh)
    DRAM = ttnn.DRAM_MEMORY_CONFIG
    shapes = [  # name, M, K, N (per die)
        ("q|k|v (8)", 8, 3072, 1024),
        ("qkv fused (12)", 12, 3072, 3072),
        ("o", 8, 1024, 3072),
        ("gate|up fused", 8, 3072, 11008),
        ("gate", 8, 3072, 5504),
        ("down", 8, 5504, 3072),
    ]
    for name, M, K, N in shapes:
        x = ttnn.from_torch(
            torch.randn(1, 1, M, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep
        )
        w = ttnn.from_torch(
            torch.randn(K, N) * 0.02,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=rep,
            memory_config=DRAM,
        )
        mb = K * N * 1.0625 / 1e6
        res = {"default": timed(mesh, lambda: ttnn.linear(x, w, compute_kernel_config=ckc, memory_config=DRAM))}
        for nc in (16, 32, 48, 64, 88, 110):
            if (N // 32) < nc:
                continue
            pc = tpc.create_matmul_1d_decode_progcfg(M, K, N, nc, grid_w=grid.x)
            try:
                res[f"1d{nc}"] = timed(
                    mesh,
                    lambda: ttnn.linear(
                        x, w, compute_kernel_config=ckc, program_config=pc, memory_config=ttnn.L1_MEMORY_CONFIG
                    ),
                )
            except Exception as e:  # noqa: BLE001
                res[f"1d{nc}"] = f"err {str(e)[:60]}"
        try:
            wd = ttnn.to_memory_config(w, tpc.create_dram_sharded_mem_config(K, N))
            act = tpc.create_activation_shard_config(K)
            for rd in (1, 2):
                pc = tpc.create_dram_sharded_matmul_program_config(M, K, N, readers=rd)
                xs = ttnn.to_memory_config(x, act)
                res[f"dsh_r{rd}"] = timed(
                    mesh,
                    lambda: ttnn.linear(
                        xs,
                        wd,
                        compute_kernel_config=ckc,
                        program_config=pc,
                        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                    ),
                )
        except Exception as e:  # noqa: BLE001
            res["dsh"] = f"err {str(e)[:80]}"
        best = min((v, k) for k, v in res.items() if isinstance(v, float))
        print(
            f"{name:16s} M{M} K{K} N{N} {mb:5.1f} MB  best {best[1]} {best[0]:.1f} us ({mb / best[0] * 1e3:.0f} GB/s)  "
            + " ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items()),
            flush=True,
        )
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
