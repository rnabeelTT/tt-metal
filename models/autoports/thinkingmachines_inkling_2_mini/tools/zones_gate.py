# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run tt/gate.py N times with the device profiler and summarize its DeviceZoneScopedN zones (G_*): per zone, the mean
duration (us at 1.35 GHz... cycles / AICLK MHz) on the root core and the max over the other cores."""

import collections
import csv
import os

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.gate import GateKernel


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    H, Rk = 2304, 64
    k = GateKernel(mesh, H, Rk, 1e-6)
    up = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh,  # noqa: E731
                                       memory_config=ttnn.L1_MEMORY_CONFIG)
    h, m = up(torch.randn(1, 1, 1, H), ttnn.float32), up(torch.randn(1, 1, 1, H), ttnn.float32)
    wd = up(torch.randn(1, 1, 2 * H, Rk) * 0.05, ttnn.bfloat16)
    wfg = up(torch.randn(1, 1, Rk, 2 * H) * 0.3, ttnn.bfloat16)
    for _ in range(int(os.environ.get("N", "10"))):
        k(h, m, wd, wfg)
    ttnn.synchronize_device(mesh)
    ttnn.ReadDeviceProfiler(mesh)
    ttnn.close_mesh_device(mesh)
    path = "/opt/tt-metal/generated/profiler/.logs/profile_log_device.csv"
    rows = list(csv.reader(open(path)))
    freq = 1350.0
    try:
        freq = float(rows[0][0].split("CHIP_FREQ[MHz]:")[1].split(",")[0].strip())
    except Exception:  # noqa: BLE001
        pass
    hdr = [c.strip() for c in rows[1]]
    ix = {n: i for i, n in enumerate(hdr)}
    root = (k.noc[0].x, k.noc[0].y)
    open_, dur = {}, collections.defaultdict(list)
    for r in rows[2:]:
        name = r[ix["zone name"]].strip() if "zone name" in ix else ""
        if not name.startswith("G_"):
            continue
        key = (int(r[ix["core_x"]]), int(r[ix["core_y"]]), r[ix["RISC processor type"]].strip(), name)
        t = int(r[ix["time[cycles since reset]"]])
        typ = r[ix["type"]].strip()
        if typ == "ZONE_START":
            open_[key] = t
        elif typ == "ZONE_END" and key in open_:
            dur[key].append((t - open_.pop(key)) / freq)
    per = collections.defaultdict(lambda: {"root": [], "other": []})
    for (x, y, risc, name), ds in dur.items():
        per[(name, risc)]["root" if (x, y) == root else "other"].append(sum(ds[1:]) / max(len(ds) - 1, 1))
    for (name, risc), v in sorted(per.items()):
        rt = max(v["root"]) if v["root"] else float("nan")
        ot = max(v["other"]) if v["other"] else float("nan")
        om = sum(v["other"]) / len(v["other"]) if v["other"] else float("nan")
        print(f"{name:14s} {risc:8s} root {rt:7.2f} us   others max {ot:7.2f} mean {om:7.2f} us", flush=True)
    print("== zones done")


if __name__ == "__main__":
    main()
