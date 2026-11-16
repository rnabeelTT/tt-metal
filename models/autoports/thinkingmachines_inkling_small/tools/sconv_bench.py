# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device time of one ShortConv.rows() call (the fused sconv kernel, R verify rows) at the K/V width and the hidden
width per die: each traced on its own and replayed, so the wall time per replay is the program's device time plus the
replay overhead (an empty-ish reference op is timed the same way for that overhead).

usage (in the image): tools/run_dev.sh python models/autoports/thinkingmachines_inkling_small/tools/sconv_bench.py [R=3]
"""

import sys
import time

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_small.tt.sconv import ShortConv


def timed(mesh, fn, reps=200):
    fn()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    dt = (time.perf_counter() - t0) / reps
    ttnn.release_trace(mesh, tid)
    return dt


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=64_000_000)
    rep = ttnn.ReplicateTensorToMesh(mesh)
    try:
        x0 = ttnn.from_torch(torch.randn(1, 1, R, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                             mesh_mapper=rep, memory_config=ttnn.L1_MEMORY_CONFIG)
        base = timed(mesh, lambda: ttnn.add(x0, x0, memory_config=ttnn.L1_MEMORY_CONFIG))
        print(f"reference op (tiny add): {1e6 * base:.1f} us per replay")
        for C in (256, 1024):
            sc = ShortConv(mesh, torch.randn(C, 4) * 0.1, rep, C)
            for mem, name in ((ttnn.L1_MEMORY_CONFIG, "L1"), (ttnn.DRAM_MEMORY_CONFIG, "DRAM")):
                x = ttnn.from_torch(torch.randn(1, 1, R, C), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                    device=mesh, mesh_mapper=rep, memory_config=mem)
                for ca in (False, True):
                    dt = timed(mesh, lambda: sc.rows(x, commit_all=ca))
                    print(f"C={C:5d} x in {name:4s} commit_all={int(ca)}: {1e6 * dt:7.1f} us per replay "
                          f"({sc.cols} cores)")
                sl = ttnn.from_torch(torch.randn(1, 1, R, 4 * C), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                     device=mesh, mesh_mapper=rep, memory_config=mem)
                # the attention path: the conv input is a column slice of the fused QKV output
                dt = timed(mesh, lambda: sc.rows(ttnn.slice(sl, (0, 0, 0, C), (1, 1, R, 2 * C),
                                                            memory_config=ttnn.L1_MEMORY_CONFIG), commit_all=False))
                print(f"C={C:5d} slice({name}) -> rows: {1e6 * dt:7.1f} us per replay")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
