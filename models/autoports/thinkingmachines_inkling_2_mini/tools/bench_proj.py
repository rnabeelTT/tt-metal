# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode (M = 32) projection matmul variants on the attention shapes, traced (dispatch gaps included like the real
step): default ttnn.linear, explicit 1D mcast_in0 on n cores (interleaved weight), DRAM-sharded with 1..3 readers per
bank; HiFi2 / LoFi; bfp8 weights, fp32 dest acc. Prints us per matmul.

usage: tools/run_dev.sh python .../tools/bench_proj.py"""

import time

import torch

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc

SHAPES = {"q|r|s|g": (2304, 6944), "k / v": (2304, 1024), "o": (3072, 2304), "gate wd": (4608, 256)}
REPS, ITERS = 20, 10


def ckc(mesh, fid):
    return ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=fid, math_approx_mode=False,
                                                  fp32_dest_acc_en=True, packer_l1_acc=True)


def timed(mesh, fn):
    fn()  # compile
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(REPS):
        fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    dt = (time.perf_counter() - t0) / (REPS * ITERS) * 1e6
    ttnn.release_trace(mesh, tid)
    return dt


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
    grid = mesh.compute_with_storage_grid_size()
    print(f"grid {grid.x}x{grid.y}", flush=True)
    for name, (K, N) in SHAPES.items():
        w = torch.randn(1, 1, K, N) * K**-0.5
        x = ttnn.from_torch(torch.randn(1, 1, 32, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                            memory_config=ttnn.L1_MEMORY_CONFIG)
        wi = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
        Np = -(-N // 256) * 256
        ws = ttnn.from_torch(torch.nn.functional.pad(w, (0, Np - N)), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
                             device=mesh, memory_config=tpc.create_dram_sharded_mem_config(K, Np))
        xs = ttnn.to_memory_config(x, tpc.create_activation_shard_config(K))
        mb = K * N * 1088 / 1024 / 1e6
        print(f"== {name}: K={K} N={N} ({mb:.1f} MB bfp8)", flush=True)
        for fname, fid in (("HiFi2", ttnn.MathFidelity.HiFi2), ("LoFi", ttnn.MathFidelity.LoFi)):
            c = ckc(mesh, fid)
            res = {}
            res["default"] = timed(mesh, lambda: ttnn.linear(x, wi, compute_kernel_config=c,
                                                             memory_config=ttnn.L1_MEMORY_CONFIG))
            for nc in (16, 24, 32, 48, 64, 80, 110):
                if nc > grid.x * grid.y:
                    continue
                pc = tpc.create_matmul_1d_decode_progcfg(32, K, N, nc, grid_w=grid.x)
                try:
                    res[f"1d{nc}"] = timed(mesh, lambda: ttnn.linear(x, wi, compute_kernel_config=c, program_config=pc,
                                                                    memory_config=ttnn.L1_MEMORY_CONFIG))
                except Exception as e:  # noqa: BLE001
                    res[f"1d{nc}"] = f"err {str(e)[:60]}"
            for rd in (1, 2, 3):
                pc = tpc.create_dram_sharded_matmul_program_config(32, K, Np, readers=rd)
                try:
                    res[f"dsh r{rd}"] = timed(mesh, lambda: ttnn.linear(
                        xs, ws, compute_kernel_config=c, program_config=pc,
                        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG))
                except Exception as e:  # noqa: BLE001
                    res[f"dsh r{rd}"] = f"err {str(e)[:60]}"
            print(f"  {fname}: " + "  ".join(f"{k} {v:.1f}" if isinstance(v, float) else f"{k} {v}"
                                             for k, v in res.items()), flush=True)
    ttnn.close_mesh_device(mesh)
    print("== bench done")


if __name__ == "__main__":
    main()
