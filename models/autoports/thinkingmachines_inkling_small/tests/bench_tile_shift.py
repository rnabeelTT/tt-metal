# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Micro-benchmark (one device, traced): row shifts of fp32 TILE tensors without row-major round trips."""
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
    C = 1024
    for R in (1, 2, 4):
        s_t, x_t = torch.randn(1, 1, 3, C), torch.randn(1, 1, R, C)
        st = ttnn.from_torch(s_t, dtype=f32, layout=TL, device=dev)
        xt = ttnn.from_torch(x_t, dtype=f32, layout=TL, device=dev)
        seq_ref = torch.cat([s_t, x_t], 2)
        tests = {
            "tile concat": (lambda: ttnn.concat([st, xt], dim=2), seq_ref),
            "tile slice j=1": (lambda: ttnn.slice(ttnn.concat([st, xt], dim=2), [0, 0, 1, 0], [1, 1, 1 + R, C]),
                               seq_ref[:, :, 1:1 + R]),
            "untilize": (lambda: ttnn.to_layout(xt, RM), x_t),
            "untilize_wu": (lambda: ttnn.untilize_with_unpadding(xt, [0, 0, R - 1, C - 1]), x_t),
            "tile mac": (lambda: ttnn.mac(xt, xt, xt), x_t * x_t + x_t),
            "rm mul": (lambda: ttnn.multiply(ttnn.to_layout(xt, RM), ttnn.to_layout(xt, RM)), None),
        }
        for name, (f, ref) in tests.items():
            try:
                got = ttnn.to_torch(f()).float()
                ok = "" if ref is None else f" max|err| {(got[..., :ref.shape[2], :] - ref).abs().max():.2e}"
                print(f"C={C} R={R} {name}: {timed(f):.1f} us{ok}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"C={C} R={R} {name}: FAILED {str(e)[:160]}", flush=True)
finally:
    ttnn.close_device(dev)
