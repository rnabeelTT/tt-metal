# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DecodeLinear (models/demos/blackhole/qwen36/tt/decode_linear/ops.py) on one chip vs torch, then traced timing vs the
shipped ttnn 1D decode matmul configs at Ornith's TP4 per-device shapes.

    python test_decode_linear.py
"""

import time

import torch

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.decode_linear.ops import DecodeLinear

SHAPES = [("gdn in-proj", 2048, 3136, 44, 11), ("attn qkv", 2048, 2560, 64, None), ("out-proj / wo", 1024, 2048, 33, 11)]


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def bench(dev, fn, reps=20, iters=20):
    ttnn.deallocate(fn())
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    for _ in range(reps):
        ttnn.deallocate(fn())
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(iters):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    ttnn.release_trace(dev, tid)
    return (time.perf_counter() - t0) / (reps * iters) * 1e6


def main():
    dev = ttnn.open_device(device_id=0, trace_region_size=64 << 20)
    try:
        ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
                                                     math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
        for name, K, N, cores1d, gw in SHAPES:
            w = torch.randn(K, N) * 0.02
            w_il = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
            wq = ttnn.to_torch(w_il).float()  # the bfp8-rounded weight both paths read
            mb = K // 32 * (-(-N // 32)) * 1088 / 1e6
            for rows in (1,):
                xt = torch.randn(1, 1, rows, K).bfloat16()
                x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                    memory_config=ttnn.L1_MEMORY_CONFIG)
                ref = xt.float()[0, 0] @ wq
                kw = {"grid_w": gw} if gw else {}
                pc = tpc.create_matmul_1d_decode_progcfg(32, K, N, cores1d, **kw)
                y1 = ttnn.linear(x, w_il, compute_kernel_config=ckc, program_config=pc, memory_config=ttnn.L1_MEMORY_CONFIG)
                p1 = pcc(ttnn.to_torch(y1).float()[0, 0, :rows], ref)
                t1 = bench(dev, lambda: ttnn.linear(x, w_il, compute_kernel_config=ckc, program_config=pc,
                                                    memory_config=ttnn.L1_MEMORY_CONFIG))
                line = f"{name} rows {rows}: ttnn 1D {t1:5.1f} us ({mb / t1 * 1e3:4.0f} GB/s, pcc {p1:.6f})"
                for readers, fid in ((4, ttnn.MathFidelity.HiFi2), (6, ttnn.MathFidelity.HiFi2), (8, ttnn.MathFidelity.HiFi2),
                                     (6, ttnn.MathFidelity.LoFi)):
                    op = DecodeLinear(dev, K, N, ttnn.bfloat8_b, readers=readers, fidelity=fid)
                    wd = op.weight([w])
                    y = op(x, wd, rows)
                    p = pcc(ttnn.to_torch(y).float()[0, 0, :rows], ref)
                    t = bench(dev, lambda: op(x, wd, rows))
                    line += f" | x{readers}{'L' if fid == ttnn.MathFidelity.LoFi else ''} {t:5.1f} us ({mb / t * 1e3:4.0f} GB/s, pcc {p:.6f})"
                    ttnn.deallocate(wd)
                print(line, flush=True)
                ttnn.deallocate(x)
            ttnn.deallocate(w_il)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
