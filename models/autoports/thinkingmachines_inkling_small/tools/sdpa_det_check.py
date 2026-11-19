# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Is inkling_sdpa's prefill launch deterministic? One die, Inkling's per-die shapes (tests/test_sdpa_device.py): the
same inputs (q, K / V cache, positions, page table, Rq, bias table) launched DET_N times, every output compared bit for
bit with the first. Cases: sliding (W = E = 512) and global (W = 0, E = 1024) layers at 512 and 2048 prefill rows
(items of 8 rows, as the model's prefill).

usage (in the image): tools/run_dev.sh python models/autoports/thinkingmachines_inkling_small/tools/sdpa_det_check.py
"""

import os

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_small.tests import test_sdpa_device as T
from models.autoports.thinkingmachines_inkling_small.tt import relbias
from models.autoports.thinkingmachines_inkling_small.tt.attn.inkling_sdpa import inkling_sdpa_decode


def case(dev, W, E, p0, n, reps, seed=0):
    kw = {"mesh_mapper": ttnn.ReplicateTensorToMesh(dev)} if isinstance(dev, ttnn.MeshDevice) else {}
    torch.manual_seed(seed)
    pos = list(range(p0, p0 + n))
    P = max(pos) + 1
    NH, NKV, HD, DREL, BLK = T.NH, T.NKV, T.HD, T.DREL, T.BLK
    proj = torch.randn(DREL, E) * 0.5
    r = torch.randn(n, NH, DREL) * 0.5
    q_h = torch.nn.functional.normalize(torch.randn(1, n, NH, HD), dim=-1) * HD**0.5
    Kp = torch.nn.functional.normalize(torch.randn(P, NKV, HD), dim=-1) * HD**0.5
    Vp = torch.randn(P, NKV, HD)
    dram = ttnn.DRAM_MEMORY_CONFIG
    nlog = (P + BLK - 1) // BLK
    if W == 0:
        nphys = nlog + 2
        pt = torch.randperm(nphys)[:nlog]
    else:
        nphys = (W + n) // BLK + 2
        pt = torch.arange(nlog) % nphys
    Kc = torch.zeros(nphys, NKV, BLK, HD)
    Vc = torch.zeros(nphys, NKV, BLK, HD)
    for p in range(P):
        b, o = divmod(p, BLK)
        Kc[pt[b], :, o] = Kp[p]
        Vc[pt[b], :, o] = Vp[p]
    K = ttnn.from_torch(Kc, dtype=T.KV_DT, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=dram, **kw)
    V = ttnn.from_torch(Vc, dtype=T.KV_DT, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=dram, **kw)
    page_table = pt[:nlog].to(torch.int32).reshape(1, -1)
    q = ttnn.from_torch(q_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=dram, **kw)
    rq_h = torch.zeros(1, n, NH, 32)
    rq_h[..., :DREL] = r
    rq = ttnn.from_torch(rq_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=dram, **kw)
    table = ttnn.from_torch(
        relbias.build_tables(proj, T.SCALE).reshape(1, 32, 32, -1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=dram,
        **kw,
    )
    cur = ttnn.from_torch(
        torch.tensor(pos, dtype=torch.int32).reshape(-1, 8),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=dev,
        memory_config=dram,
        **kw,
    )
    ptt = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=dram, **kw)
    outs = []
    for _ in range(reps):
        o = inkling_sdpa_decode(q, K, V, cur, ptt, rq, table, T.SCALE, W, E)
        outs.append(ttnn.to_torch(ttnn.get_device_tensors(o)[0] if kw else o).float())
        ttnn.deallocate(o)
    bad_rows = set()
    ndiff = 0
    for o in outs[1:]:
        d = (o - outs[0]).abs()
        if d.max() > 0:
            ndiff += 1
            bad_rows |= set(torch.nonzero(d.reshape(n, -1).amax(1)).reshape(-1).tolist())
    rows = sorted(bad_rows)
    print(
        f"W={W:3d} E={E:4d} rows {p0}..{p0 + n - 1}: {reps - 1 - ndiff} of {reps - 1} repeats identical"
        + (f"; differing rows: {len(rows)} (first {rows[:12]}, items {sorted({x // 8 for x in rows})[:12]})" if rows else ""),
        flush=True,
    )
    for t in (q, K, V, rq, table, cur, ptt):
        ttnn.deallocate(t)
    return ndiff == 0


def main():
    reps = int(os.environ.get("DET_N", "6"))
    mesh = os.environ.get("DET_MESH") == "1"  # the 1x4 mesh with fabric, as the model runs (its compute grid)
    if mesh:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576)
        print("grid", dev.compute_with_storage_grid_size(), flush=True)
    else:
        dev = ttnn.open_device(device_id=0, l1_small_size=24576)
    ok = True
    try:
        for W, E in ((512, 512), (0, 1024)):
            for n in (512, 1024, 2048):
                ok &= case(dev, W, E, 0, n, reps)
        print("RESULT", "DETERMINISTIC" if ok else "NOT DETERMINISTIC", flush=True)
    finally:
        if mesh:
            os._exit(0)
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
