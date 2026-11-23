# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Traced device time of the fused decode MoE (tt/moe_fused.py) on one layer: route + gate|up + down per call, at
ROWS rows (default 1,4,8), real routing (random normal rows). The kernels' reader knobs (QWEN35_MOE_*) are read at
construction: one process per setting.

usage: tools/run_dev.sh python .../tools/bench_moe.py     env LAYER (0), ROWS, EXPERT_DT (bfp8)"""

import os
import time

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.moe import MoE
from models.autoports.thinkingmachines_inkling_2_mini.tt.weights import Checkpoint, load_config

REPS, ITERS = 24, 10


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
    cfg, ck = load_config(), Checkpoint()
    dt = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}[os.environ.get("EXPERT_DT", "bfp8")]
    li = int(os.environ.get("LAYER", "0"))
    moe = MoE(mesh, cfg, ck, li, expert_dtype=dt, cache_dir="/cache/wcache")
    fe = moe.fe
    rows = [int(v) for v in os.environ.get("ROWS", "1,4,8").split(",")]
    fe.build_rows(rows)
    torch.manual_seed(0)
    for M in rows:
        x = ttnn.from_torch(torch.randn(1, 1, M, cfg.hidden_size).to(torch.bfloat16), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=mesh)
        fe.forward(x)
        ttnn.synchronize_device(mesh)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        for _ in range(REPS):
            fe.forward(x)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(ITERS):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        us = (time.perf_counter() - t0) / (REPS * ITERS) * 1e6
        ttnn.release_trace(mesh, tid)
        print(f"MoE layer {li} M={M}: {us:.1f} us per call", flush=True)
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
