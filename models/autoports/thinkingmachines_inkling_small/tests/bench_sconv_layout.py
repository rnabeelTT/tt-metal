# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Micro-benchmark: the short conv's tap arithmetic on row-major fp32 rows vs slice + tilize (one device)."""
import time

import torch
import ttnn

dev = ttnn.open_device(device_id=0, trace_region_size=50_000_000)
try:
    C = 4096
    for R in (1, 4):
        st = torch.randn(1, 1, 3, C)
        x = torch.randn(1, 1, R, C)
        w = torch.randn(4, C)
        ref = x * (1 + w[3])
        seq_t = torch.cat([st, x], 2)
        for j in range(3):
            ref = ref + seq_t[:, :, j:j + R] * w[j]
        RM, TL, f32 = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT, ttnn.float32
        seq = ttnn.from_torch(seq_t, dtype=f32, layout=RM, device=dev)
        xt = ttnn.from_torch(x, dtype=f32, layout=TL, device=dev)
        taps_t = [ttnn.from_torch(w[j].reshape(1, 1, 1, C), dtype=f32, layout=TL, device=dev) for j in range(4)]
        taps_r = [ttnn.from_torch(w[j].reshape(1, 1, 1, C), dtype=f32, layout=RM, device=dev) for j in range(4)]

        def cur():
            y = None
            for j in range(3):
                sl = ttnn.to_layout(ttnn.slice(seq, [0, 0, j, 0], [1, 1, j + R, C]), TL)
                y = ttnn.multiply(sl, taps_t[j]) if y is None else ttnn.mac(sl, taps_t[j], y)
            y = ttnn.mac(xt, taps_t[3], y)
            return ttnn.add(y, xt)

        def rm():
            xr = ttnn.slice(seq, [0, 0, 3, 0], [1, 1, 3 + R, C])
            y = ttnn.multiply(xr, taps_r[3])
            y = ttnn.add(y, xr)
            for j in range(3):
                sl = ttnn.slice(seq, [0, 0, j, 0], [1, 1, j + R, C])
                y = ttnn.add(y, ttnn.multiply(sl, taps_r[j]))
            return ttnn.to_layout(y, TL)

        for name, f in (("slice+tilize (current)", cur), ("row-major math", rm)):
            try:
                out = f()
                got = ttnn.to_torch(out).float()[:, :, :R]
                tid = ttnn.begin_trace_capture(dev, cq_id=0)
                f()
                ttnn.end_trace_capture(dev, tid, cq_id=0)
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
                t = time.perf_counter()
                for _ in range(50):
                    ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(dev)
                ttnn.release_trace(dev, tid)
                dt = (time.perf_counter() - t) / 50
                print(f"R={R} {name}: {dt * 1e6:.0f} us/call (host+device), max|err| {(got - ref).abs().max():.3e}",
                      flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"R={R} {name}: FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
finally:
    ttnn.close_device(dev)
