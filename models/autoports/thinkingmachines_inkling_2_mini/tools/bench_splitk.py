# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Small-N decode matmuls (one tile row): 1D mcast (tt/dmm.py) vs split-K as a batched matmul ([1, 1, 32, K] viewed as
[1, P, 32, K / P] against [1, P, K / P, N], then sum over P), traced; us per matmul and max diff."""

import time

import torch

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc

REPS, ITERS = 20, 10
SHAPES = {"gate down": (4608, 256, ttnn.bfloat16), "mix logits nq=4": (2304, 128, ttnn.float32),
          "mix logits nq=1": (2304, 32, ttnn.float32), "router": (2304, 128, ttnn.float32)}


def timed(mesh, fn):
    out = fn()
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
    return dt, out


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
    grid = mesh.compute_with_storage_grid_size()
    ckc = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
    L1 = ttnn.L1_MEMORY_CONFIG
    for name, (K, N, wdt) in SHAPES.items():
        xt, wt = torch.randn(1, 1, 32, K), torch.randn(1, 1, K, N) * K**-0.5
        x = ttnn.from_torch(xt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=L1)
        w = ttnn.from_torch(wt, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=mesh)
        nt = N // 32
        pc = tpc.create_matmul_1d_decode_progcfg(32, K, N, min(48, nt), grid_w=grid.x)
        t, ref = timed(mesh, lambda: ttnn.matmul(x, w, compute_kernel_config=ckc, program_config=pc, memory_config=L1,
                                                 dtype=ttnn.float32))
        ref = ttnn.to_torch(ref).float()
        res = [f"1d {t:.1f}"]
        for P in (4, 8, 16, 24):
            if (K // 32) % P:
                continue
            wp = ttnn.from_torch(wt.reshape(K // P, P, N).transpose(0, 1).reshape(1, P, K // P, N) if False else
                                 wt.reshape(1, P, K // P, N), dtype=wdt, layout=ttnn.TILE_LAYOUT, device=mesh)
            try:
                def f():
                    xp = ttnn.reshape(x, (1, P, 32, K // P))
                    y = ttnn.matmul(xp, wp, compute_kernel_config=ckc, memory_config=L1, dtype=ttnn.float32)
                    return ttnn.sum(y, dim=1, keepdim=True)
                tp_, o = timed(mesh, f)
                d = (ttnn.to_torch(o).float().reshape(ref.shape) - ref).abs().max().item()
                res.append(f"P{P} {tp_:.1f} (diff {d:.1e})")
            except Exception as e:  # noqa: BLE001
                res.append(f"P{P} err {str(e)[:80]}")
        print(f"{name} K={K} N={N}: " + "  ".join(res), flush=True)
    ttnn.close_mesh_device(mesh)
    print("== bench done")


if __name__ == "__main__":
    main()
