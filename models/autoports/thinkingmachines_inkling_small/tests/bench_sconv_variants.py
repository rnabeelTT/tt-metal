# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Micro-benchmark (one device, traced): whole short-conv rows() variants, exactness vs fp32 torch."""
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
    for C in (4096, 1024):
        w = torch.randn(4, C)
        taps_t = [ttnn.from_torch(w[j].reshape(1, 1, 1, C), dtype=f32, layout=TL, device=dev) for j in range(4)]
        taps_r = [ttnn.from_torch(w[j].reshape(1, 1, 1, C), dtype=f32, layout=RM, device=dev) for j in range(4)]
        st_t = torch.randn(1, 1, 3, C)
        st = ttnn.from_torch(st_t, dtype=f32, layout=RM, device=dev)
        for R in (1, 2, 4):
            x_t = torch.randn(1, 1, R, C)
            ref = x_t * (1 + w[3])
            seq_t = torch.cat([st_t, x_t], 2)
            for j in range(3):
                ref = ref + seq_t[:, :, j:j + R] * w[j]
            xt = ttnn.from_torch(x_t, dtype=f32, layout=TL, device=dev)

            def v0():  # current
                seq = ttnn.concat([st, ttnn.to_layout(xt, RM)], dim=2)
                y = None
                for j in range(3):
                    sl = ttnn.to_layout(ttnn.slice(seq, [0, 0, j, 0], [1, 1, j + R, C]), TL)
                    y = ttnn.multiply(sl, taps_t[j]) if y is None else ttnn.mac(sl, taps_t[j], y)
                y = ttnn.mac(xt, taps_t[3], y)
                return ttnn.add(y, xt)

            def rm_math(seq):
                xr = ttnn.slice(seq, [0, 0, 3, 0], [1, 1, 3 + R, C])
                y = ttnn.add(ttnn.multiply(xr, taps_r[3]), xr)
                for j in range(3):
                    y = ttnn.add(y, ttnn.multiply(ttnn.slice(seq, [0, 0, j, 0], [1, 1, j + R, C]), taps_r[j]))
                return y

            def v1():
                seq = ttnn.concat([st, ttnn.to_layout(xt, RM)], dim=2)
                return ttnn.to_layout(rm_math(seq), TL)

            def v2():
                seq = ttnn.concat([st, ttnn.untilize_with_unpadding(xt, [0, 0, R - 1, C - 1])], dim=2)
                return ttnn.tilize_with_val_padding(rm_math(seq), [1, 1, 32, C], 0.0, use_multicore=True)

            def v3():
                seq = ttnn.concat([st, ttnn.untilize_with_unpadding(xt, [0, 0, R - 1, C - 1])], dim=2)
                return ttnn.tilize_with_val_padding(rm_math(seq), [1, 1, 32, C], 0.0, use_multicore=False)

            def v4():  # tile slices via tilize_wvp, tile math
                seq = ttnn.concat([st, ttnn.untilize_with_unpadding(xt, [0, 0, R - 1, C - 1])], dim=2)
                y = None
                for j in range(3):
                    sl = ttnn.tilize_with_val_padding(ttnn.slice(seq, [0, 0, j, 0], [1, 1, j + R, C]),
                                                      [1, 1, 32, C], 0.0, use_multicore=True)
                    y = ttnn.multiply(sl, taps_t[j]) if y is None else ttnn.mac(sl, taps_t[j], y)
                y = ttnn.mac(xt, taps_t[3], y)
                return ttnn.add(y, xt)

            for name, f in (("v0 current", v0), ("v1 rm+to_layout", v1), ("v2 rm+tilize_mc", v2),
                            ("v3 rm+tilize_sc", v3), ("v4 tile+tilize_mc", v4)):
                try:
                    got = ttnn.to_torch(f()).float()[:, :, :R]
                    err = (got - ref).abs().max().item()
                    print(f"C={C} R={R} {name}: {timed(f):.1f} us  max|err| {err:.2e}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"C={C} R={R} {name}: FAILED {str(e)[:200]}", flush=True)
finally:
    ttnn.close_device(dev)
