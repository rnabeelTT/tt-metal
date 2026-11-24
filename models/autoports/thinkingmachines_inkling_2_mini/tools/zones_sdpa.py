# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode SDPA (tt/attn/inkling_sdpa.py) device zones: one decode row of a sliding (W 1024) or global layer at position
POS, run N times with the device profiler; per zone (S_*, and the RISC kernels): mean us, max over cores.
env: TP (1: 24 q / 8 kv heads), POS (300), W (1024; 0 = global), N (6), B (0: one decode row; > 0: B users at POS,
POS + 7, .. as the batched decoder calls it); the CSV under TT_METAL_PROFILER_DIR when set (a private profiler dir)"""

import collections
import csv
import glob
import os
import sys

import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import test_sdpa_device as T  # noqa: E402


def main():
    dev = ttnn.open_device(device_id=0)
    pos, W, B = int(os.environ.get("POS", "300")), int(os.environ.get("W", "1024")), int(os.environ.get("B", "0"))
    for i in range(int(os.environ.get("N", "6"))):
        if B:
            T.run_case(dev, W, T.E, [pos + 7 * u for u in range(B)], seed=i, users=B)
        else:
            T.run_case(dev, W, T.E, [pos], seed=i)
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)
    ttnn.close_device(dev)
    path = "/opt/tt-metal/generated/profiler/.logs/profile_log_device.csv"
    if os.environ.get("TT_METAL_PROFILER_DIR"):
        path = glob.glob(os.environ["TT_METAL_PROFILER_DIR"] + "/**/profile_log_device.csv", recursive=True,
                         include_hidden=True)[0]
    rows = list(csv.reader(open(path)))
    freq = 1350.0
    try:
        freq = float(rows[0][0].split("CHIP_FREQ[MHz]:")[1].split(",")[0].strip())
    except Exception:  # noqa: BLE001
        pass
    hdr = [c.strip() for c in rows[1]]
    ix = {n: i for i, n in enumerate(hdr)}
    open_, dur = {}, collections.defaultdict(list)
    for r in rows[2:]:
        name = r[ix["zone name"]].strip()
        if not (name.startswith("S_") or name in ("TRISC-KERNEL", "BRISC-KERNEL", "NCRISC-KERNEL")):
            continue
        key = (int(r[ix["core_x"]]), int(r[ix["core_y"]]), r[ix["RISC processor type"]].strip(), name)
        t = int(r[ix["time[cycles since reset]"]])
        if r[ix["type"]].strip() == "ZONE_START":
            open_[key] = t
        elif key in open_:
            dur[key].append((t - open_.pop(key)) / freq)
    per = collections.defaultdict(list)
    for (_, _, risc, name), ds in dur.items():
        per[(name, risc)].append(sum(ds[1:]) / max(len(ds) - 1, 1))
    for (name, risc), v in sorted(per.items()):
        print(f"{name:14s} {risc:8s} max {max(v):7.2f} mean {sum(v) / len(v):7.2f} us ({len(v)} cores)", flush=True)
    print("== zones done")


if __name__ == "__main__":
    main()
