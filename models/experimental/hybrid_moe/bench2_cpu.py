# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Hybrid-MoE POC, microbenchmark 2: host CPU side (Ryzen 7 9700X, 2ch DDR5-3600).

(1) Raw DDR read bandwidth (torch reduction over a 4 GiB tensor).
(2) One routed expert's SwiGLU at decode (M=1) for 397B shapes (H=4096, N=1024), weights cycling over a pool far
    larger than L3 so every call streams from DRAM. Formats: bf16 (2 B/param), int8 (1 B), int4 (0.5 B + scales).
Reports effective GB/s of weight bytes and us per expert.
Optional argv[1]: seconds to keep looping the int4 expert kernel (contention test), printing GB/s per second.
"""

import json
import sys
import time

import torch

H, N = 4096, 1024
POOL_BYTES = 2e9


def ddr_read_bw(threads):
    torch.set_num_threads(threads)
    x = torch.ones(1 << 30, dtype=torch.float32)  # 4 GiB
    x.sum()
    t0 = time.perf_counter()
    for _ in range(5):
        x.sum()
    return 5 * x.numel() * 4 / (time.perf_counter() - t0) / 1e9


def make_pool(fmt):
    bytes_per = {"bf16": 2, "int8": 1, "int4": 0.5}[fmt] * 3 * H * N
    n = max(8, int(POOL_BYTES // bytes_per))
    pool = []
    for _ in range(n):
        if fmt == "bf16":
            pool.append((torch.randn(2 * N, H, dtype=torch.bfloat16), torch.randn(H, N, dtype=torch.bfloat16)))
        elif fmt == "int8":
            pool.append(
                (
                    torch.randint(-127, 127, (2 * N, H), dtype=torch.int8),
                    torch.rand(2 * N, dtype=torch.bfloat16),
                    torch.randint(-127, 127, (H, N), dtype=torch.int8),
                    torch.rand(H, dtype=torch.bfloat16),
                )
            )
        else:
            gs = 32

            def pack(nn, kk):
                w = torch.randint(0, 16, (nn, kk), dtype=torch.int32)
                wp = torch._convert_weight_to_int4pack_for_cpu(w, 2)
                sz = torch.rand(kk // gs, nn, 2, dtype=torch.bfloat16)
                return wp, sz

            pool.append((*pack(2 * N, H), *pack(H, N)))
    return pool, bytes_per


def expert(fmt, x, p):
    if fmt == "bf16":
        gu = x @ p[0].T
        g, u = gu[:, :N], gu[:, N:]
        return (torch.nn.functional.silu(g) * u) @ p[1].T
    if fmt == "int8":
        gu = torch._weight_int8pack_mm(x, p[0], p[1])
        g, u = gu[:, :N], gu[:, N:]
        return torch._weight_int8pack_mm(torch.nn.functional.silu(g) * u, p[2], p[3])
    gu = torch._weight_int4pack_mm_for_cpu(x, p[0], 32, p[1])
    g, u = gu[:, :N], gu[:, N:]
    return torch._weight_int4pack_mm_for_cpu(torch.nn.functional.silu(g) * u, p[2], 32, p[3])


def bench(fmt, threads, seconds=None):
    torch.set_num_threads(threads)
    pool, bpe = make_pool(fmt)
    x = torch.randn(1, H, dtype=torch.bfloat16)
    for p in pool[:4]:
        expert(fmt, x, p)
    if seconds:
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            t0, n = time.perf_counter(), 0
            while time.perf_counter() - t0 < 1.0:
                expert(fmt, x, pool[n % len(pool)])
                n += 1
            dt = time.perf_counter() - t0
            print(json.dumps({"t": round(time.time(), 1), "GB_per_s": round(n * bpe / dt / 1e9, 2)}), flush=True)
        return None
    reps = 3 * len(pool)
    t0 = time.perf_counter()
    for i in range(reps):
        expert(fmt, x, pool[i % len(pool)])
    dt = time.perf_counter() - t0
    return {"GB_per_s": round(reps * bpe / dt / 1e9, 2), "us_per_expert": round(dt / reps * 1e6, 1)}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        bench("int4", 8, seconds=float(sys.argv[1]))
        sys.exit(0)
    res = {f"ddr_read_GBps_{t}t": round(ddr_read_bw(t), 2) for t in (8, 16)}
    print(res, flush=True)
    for fmt in ("bf16", "int8", "int4"):
        for t in (8, 16):
            res[f"{fmt}_{t}t"] = bench(fmt, t)
            print(fmt, t, res[f"{fmt}_{t}t"], flush=True)
    json.dump(res, open("bench2.json", "w"), indent=1)
