# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt/ccl.py's packed decode all-reduce (I2M_AR_PACK) vs the all_reduce_async path on the same per-die partials:
T = 1, 2, 4, 8, 16 live rows of [1, 1, T, 2304] bf16; bitwise on the live rows, plus traced us per all-reduce for both.
usage: MESH=4 tools/run_dev.sh python .../tests/test_arpack_device.py"""

import os
import time

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt import ccl


def main():
    N = int(os.environ.get("MESH", "4"))
    H = 2304
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, N), trace_region_size=64 << 20)
    ccl.PACK = True
    ccl.prepare(mesh, H)
    ok = True

    def timed(fn):
        fn()
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        for _ in range(20):
            o = fn()
            ttnn.deallocate(o)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(10):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        ttnn.release_trace(mesh, tid)
        return (time.perf_counter() - t0) / 200 * 1e6

    for T in [int(v) for v in os.environ.get("TS", "1,2,4,8,16").split(",")]:
        parts = torch.randn(N, 1, T, H) * 4
        x = ttnn.from_torch(parts.permute(1, 2, 0, 3).reshape(1, 1, T, N * H), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=mesh,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG if os.environ.get("DRAM", "1") == "1" else
                            ttnn.L1_MEMORY_CONFIG,
                            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3))
        outs = {}
        for pk in (True, False):
            ccl.PACK = pk
            res = []
            for site in ("mixer", "mlp", "mixer", "mlp"):
                r = ccl.all_reduce(x, mesh, site=site)
                res.append([ttnn.to_torch(t).float()[0, 0, :T] for t in ttnn.get_device_tensors(r)])
                ttnn.deallocate(r)
            outs[pk] = [v for rr in res for v in rr]
        same = all(torch.equal(a, b) for a, b in zip(outs[True], outs[False]))
        agree = all(torch.equal(outs[True][0], o) for o in outs[True])
        if not same:
            print('  per call max|packed - async|:', [round((a - b).abs().max().item(), 3) for a, b in zip(outs[True], outs[False])])
        ok &= same and agree
        us = {}
        for pk in (True, False):
            ccl.PACK = pk
            us[pk] = timed(lambda: ccl.all_reduce(x, mesh, site="mlp"))
        print(f"T={T:2d}: packed == async {same}, dies agree {agree}; packed {us[True]:.1f} us vs async {us[False]:.1f} us",
              flush=True)
    print("PASS" if ok else "FAIL")
    from models.autoports.thinkingmachines_inkling_2_mini.tt.clean_exit import close

    close(mesh)


if __name__ == "__main__":
    main()
