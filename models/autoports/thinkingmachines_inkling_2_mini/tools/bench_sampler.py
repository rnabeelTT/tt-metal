# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device sampler ops (tt/sampling.py) on [1, 1, W, Vp] fp32 logits, traced: typecast, RowsTopK, gather, slices."""

import time

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.decode_sampling.ops import RowsTopK

VP, REPS, ITERS = 200064, 10, 10


def timed(mesh, fn):
    fn()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(REPS):
        fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    ttnn.release_trace(mesh, tid)
    return (time.perf_counter() - t0) / (REPS * ITERS) * 1e3


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=128 << 20)
    rtk = RowsTopK(mesh, VP)
    from models.autoports.thinkingmachines_inkling_2_mini.tt.sampling import DeviceTopK

    dtk = DeviceTopK(mesh, VP, VP - 6)
    for W in (1, 8):
        lg = ttnn.from_torch(torch.randn(1, 1, W, VP) * 5, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
        lb = ttnn.typecast(lg, ttnn.bfloat16)
        _, idx = rtk(lb, W)
        idx = ttnn.slice(idx, [0, 0, 0, 0], [1, 1, W, 32])
        r = {"typecast": timed(mesh, lambda: ttnn.typecast(lg, ttnn.bfloat16)),
             "rowstopk": timed(mesh, lambda: rtk(lb, W)),
             "slice idx": timed(mesh, lambda: ttnn.slice(idx, [0, 0, 0, 0], [1, 1, W, 32])),
             "gather": timed(mesh, lambda: ttnn.gather(lg, dim=-1, index=idx)),
             "two-pass (tt/sampling.py)": timed(mesh, lambda: dtk.device(lg, W)),
             "max": timed(mesh, lambda: ttnn.max(lg, dim=-1, keepdim=True)),
             "max+sub+bf16": timed(mesh, lambda: ttnn.typecast(ttnn.subtract(lg, ttnn.max(lg, dim=-1, keepdim=True)),
                                                                ttnn.bfloat16))}
        print(f"W={W}: " + "  ".join(f"{k} {v:.3f} ms" for k, v in r.items()), flush=True)
    ttnn.close_mesh_device(mesh)
    print("== bench done")


if __name__ == "__main__":
    main()
