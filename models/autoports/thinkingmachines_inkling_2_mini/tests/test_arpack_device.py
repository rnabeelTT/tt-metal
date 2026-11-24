# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt/ccl.py's packed decode all-reduce (I2M_AR_PACK; fused: I2M_AR_FUSED, one program) vs the all_reduce_async path on
the same per-die partials: T = 1, 2, 4, 8, 16 live rows of [1, 1, T, 2304] bf16; bitwise on the live rows, plus traced us
per all-reduce (fused / packed: the packed result, as post / pre read it).
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

    def timed(fn):  # (fn(site); the sites alternate, as in the decode)
        ttnn.deallocate(fn("mixer"))
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        for i in range(20):
            ttnn.deallocate(fn(("mlp", "mixer")[i % 2]))
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
        modes = {"fused": (True, True), "packed": (True, False), "async": (False, False)}
        outs = {}
        for m, (pk, fu) in modes.items():
            ccl.PACK, ccl.FUSED = pk, fu
            res = []
            for site in ("mixer", "mlp", "mixer", "mlp"):
                r = ccl.all_reduce(x, mesh, site=site)
                res.append([ttnn.to_torch(t).float()[0, 0, :T] for t in ttnn.get_device_tensors(r)])
                ttnn.deallocate(r)
            outs[m] = [v for rr in res for v in rr]
        ref = outs["async"]
        exact = torch.allclose(ref[0], parts.sum(0).float(), rtol=0.02, atol=0.1)
        msg = []
        for m in ("fused", "packed"):
            same = all(torch.equal(a, b) for a, b in zip(outs[m], ref))
            if not same:
                print(f"  {m}: per call max|{m} - async|:", [round((a - b).abs().max().item(), 3) for a, b in zip(outs[m], ref)])
            ok &= same
            msg.append(f"{m} == async {same}")
        agree = all(torch.equal(ref[0], o) for o in outs["fused"])
        ok &= agree and exact
        us = {}
        for m, (pk, fu) in modes.items():
            ccl.PACK, ccl.FUSED = pk, fu
            us[m] = timed(lambda site: (lambda r: r.t if isinstance(r, ccl.PackedAR) else r)(
                ccl.all_reduce(x, mesh, site=site, packed_out=pk)))
        print(f"T={T:2d}: {', '.join(msg)}, dies agree {agree}, ~sum {exact}; "
              + ", ".join(f"{m} {v:.1f} us" for m, v in us.items()), flush=True)
    print("PASS" if ok else "FAIL")
    from models.autoports.thinkingmachines_inkling_2_mini.tt.clean_exit import close

    close(mesh)


if __name__ == "__main__":
    main()
