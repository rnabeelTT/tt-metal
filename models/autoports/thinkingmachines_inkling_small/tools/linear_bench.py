# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode-row projections at Inkling's per-die shapes (3 verify rows padded to a tile), one chip, trace-replay time per
call: ttnn.linear as the model calls it (no program config), ttnn.linear with 1D multicast program configs, and
qwen36's DecodeLinear (bank-streamed weight copy). Also checks each variant's output against the default's.

usage (in the image): tools/run_dev.sh python models/autoports/thinkingmachines_inkling_small/tools/linear_bench.py
"""

import os
import sys
import time

import torch
import ttnn

from models.demos.blackhole.qwen36.tt.decode_linear.ops import DecodeLinear

SHAPES = {  # name: (K, N, weight dtype)
    "qkvr": (4096, 1664, ttnn.bfloat8_b),
    "o_proj": (1024, 4096, ttnn.bfloat8_b),
}
ROWS = 3


def main():
    dev = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=64 << 20)
    grid = dev.compute_with_storage_grid_size()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def bench(fn, reps=20, iters=20):
        o = fn()
        ttnn.deallocate(o)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        for _ in range(reps):
            ttnn.deallocate(fn())
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        t0 = time.time()
        for _ in range(iters):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        ttnn.release_trace(dev, tid)
        return (time.time() - t0) / (reps * iters) * 1e6

    try:
        g = torch.Generator().manual_seed(0)
        for name, (K, N, dt) in SHAPES.items():
            w_t = torch.randn(K, N, generator=g) * 0.02
            x_t = torch.zeros(1, 1, 32, K)
            x_t[0, 0, :ROWS] = torch.randn(ROWS, K, generator=g)
            w = ttnn.from_torch(w_t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            x = ttnn.from_torch(x_t[:, :, :ROWS], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                memory_config=ttnn.L1_MEMORY_CONFIG)
            ref = ttnn.to_torch(ttnn.linear(x, w, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG)).float()
            res = []
            t = bench(lambda: ttnn.linear(x, w, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG))
            res.append(("ttnn.linear default", t, 0.0))
            kt, nt = K // 32, N // 32
            for ncores in sorted({c for c in (nt, nt // 2, nt // 4, 64, 96, 110) if 0 < c <= grid.x * grid.y and nt % c == 0}):
                for bw in (2, 4, 8):
                    if kt % bw:
                        continue
                    pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                        compute_with_storage_grid_size=(grid.x, grid.y), in0_block_w=bw, out_subblock_h=1,
                        out_subblock_w=1, per_core_M=1, per_core_N=nt // ncores, fuse_batch=True,
                        fused_activation=None, mcast_in0=True,
                    )
                    try:
                        fn = lambda: ttnn.linear(x, w, compute_kernel_config=ckc, program_config=pc,  # noqa: E731
                                                 memory_config=ttnn.L1_MEMORY_CONFIG)
                        out = ttnn.to_torch(fn()).float()
                        t = bench(fn)
                        res.append((f"1D mcast {ncores} cores bw {bw}", t, (out - ref).abs().max().item()))
                    except Exception as e:  # noqa: BLE001
                        res.append((f"1D mcast {ncores} cores bw {bw}: {str(e).splitlines()[0][:80]}", float("nan"), 0))
            for readers in (2, 3, 4, 6, 8):
                try:
                    dl = DecodeLinear(dev, K, N, dt, readers=readers, fidelity=ttnn.MathFidelity.HiFi2)
                    wd = dl.weight([w_t.to(torch.bfloat16)])
                    fn = lambda: dl(x, wd, ROWS)  # noqa: E731
                    out = ttnn.to_torch(fn()).float()
                    t = bench(fn)
                    res.append((f"DecodeLinear readers {readers} ({len(dl.assign)} cores)", t,
                                (out[..., :ROWS, :] - ref[..., :ROWS, :]).abs().max().item()))
                    ttnn.deallocate(wd)
                except Exception as e:  # noqa: BLE001
                    res.append((f"DecodeLinear readers {readers}: {str(e).splitlines()[0][:80]}", float("nan"), 0))
            mb = K * N * (1088 if dt == ttnn.bfloat8_b else 576) / 1024 / 1e6
            print(f"== {name}: K {K} N {N} {dt}, {mb:.1f} MB of weight", flush=True)
            for label, t, d in res:
                print(f"   {t:7.1f} us  {mb / max(t, 1e-9) * 1e6 / 1e3:6.0f} GB/s  max|d| {d:.3g}  {label}", flush=True)
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
