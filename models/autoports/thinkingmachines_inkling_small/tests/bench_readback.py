# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Micro-benchmark (one device): host readback of [1, 1, R, V] logits, tile vs row-major sliced vs device argmax."""
import time

import torch
import ttnn

dev = ttnn.open_device(device_id=0)
import os
V = int(os.environ.get('BENCH_V', '201024'))
try:
    for R in (1, 4):
        for dt in (ttnn.bfloat16, ttnn.float32):
            lg = ttnn.from_torch(torch.randn(1, 1, R, V), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)

            def tile():
                return ttnn.to_torch(lg).float().reshape(-1, V)[:R]

            def rm():
                return ttnn.to_torch(ttnn.untilize_with_unpadding(lg, [0, 0, R - 1, V - 1])).float().reshape(-1, V)

            def amax():
                return ttnn.to_torch(ttnn.argmax(ttnn.untilize_with_unpadding(lg, [0, 0, R - 1, V - 1]), dim=-1))

            for name, f in (("tile to_torch", tile), ("untilize+to_torch", rm), ("device argmax", amax)):
                try:
                    f()
                    ttnn.synchronize_device(dev)
                    t = time.perf_counter()
                    for _ in range(20):
                        out = f()
                    dtm = (time.perf_counter() - t) / 20
                    print(f"R={R} {dt} {name}: {dtm * 1e3:.2f} ms", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"R={R} {dt} {name}: FAILED {str(e)[:200]}", flush=True)
finally:
    ttnn.close_device(dev)
