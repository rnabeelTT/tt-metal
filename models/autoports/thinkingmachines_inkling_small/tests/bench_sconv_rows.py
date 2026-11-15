# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Micro-benchmark (one device, traced): ShortConv.rows + commit as the verify pass runs them (R rows, C channels per
die: 1024 for the hidden-size convs at TP4, 256 for K / V), us per call. usage: python bench_sconv_rows.py"""
import time

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_small.tt.sconv import ShortConv

dev = ttnn.open_device(device_id=0, trace_region_size=100_000_000)


def timed(f, n=200):
    f()
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    f()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    t = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    ttnn.release_trace(dev, tid)
    return (time.perf_counter() - t) / n * 1e6


try:
    for C in (1024, 256):
        for R in (1, 3, 4):
            sc = ShortConv(dev, torch.randn(C, 4), None, C)
            x = ttnn.from_torch(torch.randn(1, 1, R, C), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

            def f(sc=sc, x=x):
                y = sc.rows(x, commit_all=False)
                sc.commit_rows(1)
                return y

            def add_only(x=x):
                return ttnn.add(x, x)

            print(f"C {C} R {R}: rows+commit {timed(f):.1f} us | one ttnn.add {timed(add_only):.1f} us", flush=True)
finally:
    ttnn.close_device(dev)
