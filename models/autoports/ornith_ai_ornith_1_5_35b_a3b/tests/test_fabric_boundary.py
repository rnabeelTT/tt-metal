# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fabric decode boundary (models/demos/blackhole/qwen36/tt/fabric_boundary/ops.py) on the (1, 4) mesh vs torch:
an entry boundary, then boundaries alternating the two sites with distinct per-device partial sums; residual' must be
bit-identical on every device and match the FP32 reference to BF16 rounding, x to the RMSNorm; rows 1..31 zero. The
same chain eager and traced.

    python test_fabric_boundary.py [ring|linear] [prof]
"""

import sys
import time

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.fabric_boundary.ops import FabricBoundary

H, EPS, ND = 2048, 1e-6, 4


MESH = []


def per_device(t):
    mesh = MESH[0]
    return list(ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float().unbind(0))


def flat_of(t, rows=1):
    """Per-device [rows, H] of a flat [1, 1, 32, 32 * pages] tensor (row r's vector at pages r * H / 1024 ..)."""
    out = []
    for d in per_device(t):
        m = d.reshape(32, -1)
        pages = m.shape[-1] // 32
        out.append(m.reshape(2, 16, pages, 2, 16).permute(2, 0, 3, 1, 4).reshape(rows, -1)[:, :H])
    return out


def rows_of(t):
    return [ttnn.to_torch(ttnn.to_layout(t, ttnn.TILE_LAYOUT), mesh_composer=ttnn.ConcatMeshToTensor(MESH[0], dim=0)).float()[i].reshape(-1, H) for i in range(ND)]


def rmsnorm(v, g):
    return v * torch.rsqrt(v.pow(2).mean() + EPS) * g


def main():
    ring = "linear" not in sys.argv
    prof = "prof" in sys.argv
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING if ring else ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, ND), trace_region_size=32 << 20)
    MESH.append(mesh)
    try:
        fb = FabricBoundary(mesh, H, EPS, ring_topology=ring)
        rep = ttnn.ReplicateTensorToMesh(mesh)
        gw = [1 + 0.1 * torch.randn(H) for _ in range(3)]
        gam = [fb.gamma(ttnn.from_torch(g.reshape(1, 1, 64, 32).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                        device=mesh, mesh_mapper=rep)) for g in gw]
        gref = [g.bfloat16().float() for g in gw]
        steps = 6
        for R in (8, 2, 1):
            e = torch.randn(1, 1, R, H)
            emb = ttnn.from_torch(e, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                  memory_config=ttnn.L1_MEMORY_CONFIG, mesh_mapper=rep)
            parts_t = [(torch.randn(ND, 1, R, H) * 0.5).bfloat16() for _ in range(steps)]
            parts = [ttnn.from_torch(p, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                     memory_config=ttnn.L1_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
                     for p in parts_t]

            def chain():
                res, x = fb(emb, None, gam[0])
                outs = [(res, x)]
                for s_ in range(steps):  # every output kept alive for the check
                    res, x = fb(res, parts[s_], gam[(s_ + 1) % 3], "mixer" if s_ % 2 == 0 else "mlp")
                    outs.append((res, x))
                return outs

            def check(outs, label):
                ref = e[0, 0].bfloat16().float()
                worst = 0.0
                for s_, (res, x) in enumerate(outs):
                    if s_ > 0:
                        ref = (ref + parts_t[s_ - 1][:, 0].float().sum(0)).bfloat16().float()
                    got = flat_of(res, R)
                    for d in range(1, ND):
                        assert torch.equal(got[d], got[0]), f"{label} step {s_}: residual differs on device {d}"
                    err_r = (got[0] - ref).abs().max().item() / ref.abs().max().item()
                    xr = rows_of(x)[0]
                    xref = torch.stack([rmsnorm(ref[r], gref[s_ % 3]) for r in range(R)])
                    err_x = (xr - xref).abs().max().item() / xref.abs().max().item()
                    worst = max(worst, err_r, err_x)
                    assert err_r < 2e-2 and err_x < 2.5e-2, (label, s_, err_r, err_x)
                print(f"{label} rows {R}: {len(outs)} boundaries ok, worst rel err {worst:.2e}", flush=True)

            outs = chain()
            check(outs, "eager")
            if R != 1:
                fn = lambda: [fb(outs[0][0], parts[i % steps], gam[1], "mixer" if i % 2 == 0 else "mlp") for i in range(12)]
                fn()
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                fn()
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                t0 = time.perf_counter()
                for _ in range(50):
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                print(f"traced all-reduce rows {R}: {(time.perf_counter() - t0) / 600 * 1e6:.1f} us per boundary", flush=True)
                ttnn.release_trace(mesh, tid)
                continue
        # traced
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        outs_t = chain()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        for _ in range(3):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        check(outs_t, "traced")
        if "proftrace" in sys.argv:
            ttnn.ReadDeviceProfiler(mesh)
            ttnn.release_trace(mesh, tid)
            return
        reps = 50
        t0 = time.perf_counter()
        for _ in range(reps):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        us = (time.perf_counter() - t0) / reps / (steps + 1) * 1e6
        print(f"traced: {us:.1f} us per boundary ({steps} all-reduce boundaries + 1 entry per replay)")
        ttnn.release_trace(mesh, tid)
        # dispatch vs device: traces of 12 entry boundaries (no fabric) and 12 all-reduce boundaries
        for label, fn in (
            ("entry only", lambda: [fb(emb, None, gam[0]) for _ in range(12)]),
            ("all-reduce", lambda: [fb(outs[0][0], parts[i % steps], gam[1], "mixer" if i % 2 == 0 else "mlp") for i in range(12)]),
        ):
            fn()
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            fn()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            t0 = time.perf_counter()
            for _ in range(reps):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            print(f"traced {label}: {(time.perf_counter() - t0) / reps / 12 * 1e6:.1f} us per boundary")
            ttnn.release_trace(mesh, tid)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
