# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill (M = T rows) attention projections on one die's shards, traced: the default ttnn.linear config (today's
path) vs explicit 2D multicast configs over grid shapes / K blocks / subblocks; bfp8 weights, HiFi2, fp32 acc.
Prints us per matmul and whether the output equals the default's bitwise.

usage: tools/run_dev.sh python .../tools/bench_pfproj.py   env: TPS (default 4,2,1), TS (default 2048)"""

import itertools
import math
import os
import time

import torch

import ttnn

REPS, ITERS = 6, 5
NH, NKV, HD, DR, H = 24, 8, 128, 16, 2304


def shapes(tp):
    nh, nkv = NH // tp, NKV // tp
    fq = nh * HD + nh * DR + 32 + nh * HD
    return {"q": (H, fq, ttnn.float32), "k": (H, nkv * HD, ttnn.bfloat16), "o": (nh * HD, H, ttnn.bfloat16)}


def timed(mesh, fn):
    fn()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(REPS):
        o = fn()
        ttnn.deallocate(o)
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    dt = (time.perf_counter() - t0) / (REPS * ITERS) * 1e6
    ttnn.release_trace(mesh, tid)
    return dt


def divisors(n, cap):
    return [d for d in range(1, cap + 1) if n % d == 0]


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=512 << 20)
    grid = mesh.compute_with_storage_grid_size()
    print(f"grid {grid.x}x{grid.y}", flush=True)
    ckc = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
                                                 math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
    DRAM = ttnn.DRAM_MEMORY_CONFIG
    for tp in [int(t) for t in os.environ.get("TPS", "4,2,1").split(",")]:
        for T in [int(t) for t in os.environ.get("TS", "2048").split(",")]:
            for name, (K, N, odt) in shapes(tp).items():
                x = ttnn.from_torch(torch.randn(1, 1, T, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                    device=mesh, memory_config=DRAM)
                w = ttnn.from_torch(torch.randn(1, 1, K, N) * K**-0.5, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
                                    device=mesh, memory_config=DRAM)
                base = lambda: ttnn.linear(x, w, compute_kernel_config=ckc, dtype=odt, memory_config=DRAM)  # noqa
                ref = ttnn.to_torch(base())
                t_def = timed(mesh, base)
                print(f"== TP{tp} T={T} {name}: K={K} N={N} default {t_def:.1f} us", flush=True)
                Mt, Nt, Kt = T // 32, math.ceil(N / 32), K // 32
                res = []
                for gx, gy in itertools.product(range(grid.x - 5, grid.x + 1), range(grid.y - 3, grid.y + 1)):
                    pm, pn = math.ceil(Mt / gy), math.ceil(Nt / gx)
                    if pn * gx - Nt >= pn or pm * gy - Mt >= pm:  # (an idle column / row of cores)
                        continue
                    for bw in [d for d in divisors(Kt, 8) if d in (2, 4, 8)]:
                        # out subblock: h * w <= 4 (fp32 acc), w | pn, h | pm
                        sbs = [(h, ww) for h in (1, 2, 4) for ww in (1, 2, 4) if h * ww <= 4 and pn % ww == 0
                               and pm % h == 0]
                        sb = max(sbs, key=lambda s: (s[0] * s[1], s[1]))
                        cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                            compute_with_storage_grid_size=(gx, gy), in0_block_w=bw, out_subblock_h=sb[0],
                            out_subblock_w=sb[1], per_core_M=pm, per_core_N=pn, transpose_mcast=False,
                            fused_activation=None, fuse_batch=False)
                        fn = lambda cfg=cfg: ttnn.linear(x, w, compute_kernel_config=ckc, dtype=odt,  # noqa
                                                         memory_config=DRAM, program_config=cfg)
                        try:
                            o = ttnn.to_torch(fn())
                            t = timed(mesh, fn)
                        except Exception as e:  # noqa: BLE001
                            continue
                        res.append((t, gx, gy, pm, pn, bw, sb, torch.equal(o, ref)))
                res.sort()
                for r in res[:6]:
                    print(f"   {r[0]:8.1f} us grid {r[1]}x{r[2]} pcM {r[3]} pcN {r[4]} bw {r[5]} sb {r[6]} "
                          f"bitwise {r[7]}", flush=True)
                for t in (x, w):
                    ttnn.deallocate(t)
    print("== bench_pfproj done", flush=True)
    from models.autoports.thinkingmachines_inkling_2_mini.tt.clean_exit import close

    close(mesh)


if __name__ == "__main__":
    main()
