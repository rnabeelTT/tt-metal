# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Micro-benchmark (one device, traced): fp32 row-major [1, 1, r, C] -> TILE, by row count and op variant."""
import time

import torch
import ttnn

dev = ttnn.open_device(device_id=0, trace_region_size=50_000_000)
RM, TL, f32 = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT, ttnn.float32


def timed(f, n=100):
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
    import os

    dts = {"f32": f32, "bf16": ttnn.bfloat16}
    for C in (1024, 256):
      for dn, dt in dts.items():
        for r in (1, 2, 4):
            src = ttnn.from_torch(torch.randn(1, 1, r + 3, C), dtype=dt, layout=RM, device=dev)
            print(f"-- dtype {dn}")
            v = {
                "slice+to_layout": lambda: ttnn.to_layout(ttnn.slice(src, [0, 0, 0, 0], [1, 1, r, C]), TL),
                "slice+tilize_wvp": lambda: ttnn.tilize_with_val_padding(
                    ttnn.slice(src, [0, 0, 0, 0], [1, 1, r, C]), [1, 1, 32, C], 0.0, use_multicore=True),
                "slice only": lambda: ttnn.slice(src, [0, 0, 0, 0], [1, 1, r, C]),
                "slice+typecast_bf16+to_layout": lambda: ttnn.to_layout(
                    ttnn.typecast(ttnn.slice(src, [0, 0, 0, 0], [1, 1, r, C]), ttnn.bfloat16), TL),
                "slice+tilize_wvp+typecast_out": lambda: ttnn.tilize_with_val_padding(
                    ttnn.slice(src, [0, 0, 0, 0], [1, 1, r, C]), [1, 1, 32, C], 0.0, dtype=ttnn.bfloat16),
            }
            for name, f in v.items():
                try:
                    print(f"C={C} r={r} {name}: {timed(f):.1f} us", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"C={C} r={r} {name}: FAILED {str(e)[:200]}", flush=True)
finally:
    ttnn.close_device(dev)
