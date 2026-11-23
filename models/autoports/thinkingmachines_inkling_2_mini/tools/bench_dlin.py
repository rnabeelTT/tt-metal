# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One-die attention projection shapes, traced: this port's decode matmul (tt/dmm.py, interleaved weight, 1D mcast)
vs the bank-streamed DecodeLinear (qwen36 decode_linear: a per-bank column-major weight copy, `readers` cores per bank),
bfp8 weights, HiFi2, one row. Prints us per matmul and GB/s.

usage: tools/run_dev.sh python .../tools/bench_dlin.py     env READERS (2,4,6)"""

import os
import time

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.dmm import dmm
from models.demos.blackhole.qwen36.tt.decode_linear.ops import DecodeLinear

SHAPES = {"q|r|s|g": (2304, 6944), "q|r|s|g|k|v": (2304, 8992), "k": (2304, 1024), "o": (3072, 2304)}
REPS, ITERS = 24, 10


def timed(mesh, fn):
    fn()
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
    us = (time.perf_counter() - t0) / (REPS * ITERS) * 1e6
    ttnn.release_trace(mesh, tid)
    return us


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
    ckc = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
                                                 math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
    torch.manual_seed(0)
    for name, (K, N) in SHAPES.items():
        wt = torch.randn(K, N) * 0.02
        xt = torch.randn(1, 1, 1, K)
        x = ttnn.from_torch(xt.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        w = ttnn.from_torch(wt.reshape(1, 1, K, N).to(torch.bfloat16), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
                            device=mesh)
        mb = K * N * 1088 / 1024 / 1e6
        ref = ttnn.to_torch(dmm(x, w, ckc, memory_config=ttnn.L1_MEMORY_CONFIG)).float().reshape(-1)[:N]
        us = timed(mesh, lambda: dmm(x, w, ckc, memory_config=ttnn.L1_MEMORY_CONFIG))
        print(f"{name:12s} K={K} N={N} ({mb:.1f} MB): dmm {us:6.1f} us ({mb / us * 1e3:.0f} GB/s)", flush=True)
        for r in [int(v) for v in os.environ.get("READERS", "2,4,6").split(",")]:
            dl = DecodeLinear(mesh, K, N, ttnn.bfloat8_b, readers=r)
            wd = dl.weight([wt])
            got = ttnn.to_torch(dl(x, wd, 1)).float().reshape(-1)[:N]
            d = (got - ref).abs().max().item()
            us = timed(mesh, lambda: dl(x, wd, 1))
            print(f"{'':12s} DecodeLinear readers {r}: {us:6.1f} us ({mb / us * 1e3:.0f} GB/s)  max|d| vs dmm {d:.3g}",
                  flush=True)
            ttnn.deallocate(wd)
        ttnn.deallocate(w)
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
