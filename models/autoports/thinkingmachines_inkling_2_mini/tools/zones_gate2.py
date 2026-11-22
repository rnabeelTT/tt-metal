# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run tt/gate.py GateKernel2 N times (T rows; NORM / ROUTER like the MLP-side gate) with the device profiler; per
DeviceZoneScopedN zone (G2_*): mean us on the root core and the max / mean over the others.
env: N (10), T (1), NORM (1), ROUTER (1)"""

import collections
import csv
import os

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.gate import GateKernel2
from models.autoports.thinkingmachines_inkling_2_mini.tt.post import PostKernel


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    H, Rk, T = 2304, 64, int(os.environ.get("T", "1"))
    pk = PostKernel(mesh, H, 1e-6)
    k = GateKernel2(pk, Rk)
    L1 = ttnn.L1_MEMORY_CONFIG
    up = lambda t, dt, mc=None: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh,  # noqa: E731
                                                memory_config=mc)
    h, m = up(torch.randn(1, 1, T, H), ttnn.float32, L1), up(torch.randn(1, 1, T, H), ttnn.float32, L1)
    wd = up(torch.randn(1, 1, 2 * H, Rk) * 0.05, ttnn.bfloat16)
    wfg = up(torch.randn(1, 1, Rk, 2 * H) * 0.3, ttnn.bfloat16)
    nw = up(torch.ones(1, 1, 32, H), ttnn.float32) if os.environ.get("NORM", "1") == "1" else None
    rt = (up(torch.randn(1, 1, H, 128) * 0.02, ttnn.bfloat16),) * 2 if os.environ.get("ROUTER", "1") == "1" else None
    for _ in range(int(os.environ.get("N", "10"))):
        k(h, m, wd, wfg, nw=nw, router=rt, dup=T == 1)
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
    root = mesh_root = None
    open_, dur = {}, collections.defaultdict(list)
    for r in rows[2:]:
        name = r[ix["zone name"]].strip()
        if not (name.startswith("G2_") or name.endswith("-KERNEL")):
            continue
        key = (int(r[ix["core_x"]]), int(r[ix["core_y"]]), r[ix["RISC processor type"]].strip(), name)
        t = int(r[ix["time[cycles since reset]"]])
        if r[ix["type"]].strip() == "ZONE_START":
            open_[key] = t
        elif key in open_:
            dur[key].append((t - open_.pop(key)) / freq)
    root = min((x, y) for x, y, _, _ in dur)
    per = collections.defaultdict(lambda: {"root": [], "other": []})
    for (x, y, risc, name), ds in dur.items():
        per[(name, risc)]["root" if (x, y) == root else "other"].append(sum(ds[1:]) / max(len(ds) - 1, 1))
    for (name, risc), v in sorted(per.items()):
        f = lambda a: max(a) if a else float("nan")  # noqa: E731
        om = sum(v["other"]) / len(v["other"]) if v["other"] else float("nan")
        print(f"{name:16s} {risc:8s} root {f(v['root']):7.2f} us   others max {f(v['other']):7.2f} mean {om:7.2f} us",
              flush=True)
    print("== zones done")


if __name__ == "__main__":
    main()
