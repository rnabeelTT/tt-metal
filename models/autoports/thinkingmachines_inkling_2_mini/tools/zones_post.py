# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run tt/post.py N times (one row, 3 blocks) with the device profiler; per DeviceZoneScopedN zone (P_*): mean us, max
over cores. env: N (10), T (1), NB (3), NORM (1)"""

import collections
import csv
import os

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.batched import BatchedConv
from models.autoports.thinkingmachines_inkling_2_mini.tt.post import PostKernel


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    H, T, NB = 2304, int(os.environ.get("T", "1")), int(os.environ.get("NB", "3"))
    k = PostKernel(mesh, H, 1e-6)
    l1 = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh,  # noqa: E731
                                       memory_config=ttnn.L1_MEMORY_CONFIG)
    att, x = l1(torch.randn(1, 1, T, H), ttnn.bfloat16), l1(torch.randn(1, 1, T, H), ttnn.bfloat16)
    conv = BatchedConv(mesh, torch.randn(H, 4) * 0.3, None, H, T)
    q = ttnn.from_torch(torch.randn(1, 1, 32, H) * 0.02, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
    blocks = [l1(torch.randn(1, 1, T, H), ttnn.float32) for _ in range(NB)]
    lg = [(l1(torch.randn(1, 1, T, 64), ttnn.float32), 1) for _ in range(NB)]
    nw = k.rows_weight(ttnn.from_torch(torch.ones(1, 1, 1, H), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                       device=mesh)) if os.environ.get("NORM", "1") == "1" else None
    for _ in range(int(os.environ.get("N", "10"))):
        k(att, x, conv, q, blocks, lg, nw=nw)
    ttnn.synchronize_device(mesh)
    ttnn.ReadDeviceProfiler(mesh)
    ttnn.close_mesh_device(mesh)
    rows = list(csv.reader(open("/opt/tt-metal/generated/profiler/.logs/profile_log_device.csv")))
    freq = 1350.0
    try:
        freq = float(rows[0][0].split("CHIP_FREQ[MHz]:")[1].split(",")[0].strip())
    except Exception:  # noqa: BLE001
        pass
    hdr = [c.strip() for c in rows[1]]
    ix = {n: i for i, n in enumerate(hdr)}
    open_, dur = {}, collections.defaultdict(list)
    for r in rows[2:]:
        name = r[ix["zone name"]].strip() if "zone name" in ix else ""
        if not (name.startswith("P_") or name in ("TRISC-KERNEL", "BRISC-KERNEL", "NCRISC-KERNEL")):
            continue
        key = (int(r[ix["core_x"]]), int(r[ix["core_y"]]), r[ix["RISC processor type"]].strip(), name)
        t = int(r[ix["time[cycles since reset]"]])
        typ = r[ix["type"]].strip()
        if typ == "ZONE_START":
            open_[key] = t
        elif typ == "ZONE_END" and key in open_:
            dur[key].append((t - open_.pop(key)) / freq)
    per = collections.defaultdict(list)
    for (x_, y_, risc, name), ds in dur.items():
        per[(name, risc)].append(sum(ds[1:]) / max(len(ds) - 1, 1))
    for (name, risc), v in sorted(per.items()):
        print(f"{name:14s} {risc:8s} max {max(v):7.2f} mean {sum(v) / len(v):7.2f} us ({len(v)} cores)", flush=True)
    print("== zones done")


if __name__ == "__main__":
    main()
