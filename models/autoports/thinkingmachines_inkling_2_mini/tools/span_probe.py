# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device span of one glue program (OP=post|gate, as glz_timeline.py builds it) run N times on one die: per run,
first kernel start -> last kernel end over all cores (us), median of runs 2..N. For grid / reduction-shape probes
(I2M_POST_GRID etc.). env: N (12), T (1), OP (post)"""

import csv
import os
import statistics

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tools.glz_timeline import run_gate, run_post


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    n, T = int(os.environ.get("N", "12")), int(os.environ.get("T", "1"))
    {"gate": run_gate, "post": run_post}[os.environ.get("OP", "post")](mesh, n, T)
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
    ix = {nm: i for i, nm in enumerate(hdr)}
    ev = sorted((int(r[ix["time[cycles since reset]"]]), r[ix["zone name"]].strip(), r[ix["type"]].strip(),
                 int(r[ix["core_x"]]), int(r[ix["core_y"]])) for r in rows[2:])
    ev = [e for e in ev if e[1].endswith("-KERNEL")]
    # runs: windows between consecutive BRISC kernel starts of the root core (the lowest x, y)
    root = min((e[3], e[4]) for e in ev if e[1] == "BRISC-KERNEL")
    st = [e[0] for e in ev if (e[3], e[4]) == root and e[1] == "BRISC-KERNEL" and e[2] == "ZONE_START"] + [1 << 62]
    runs = [[e for e in ev if st[i] - 5000 <= e[0] < st[i + 1] - 5000] for i in range(len(st) - 1)]
    spans = [(max(e[0] for e in r if e[2] == "ZONE_END") - min(e[0] for e in r if e[2] == "ZONE_START")) / freq
             for r in runs if any(e[2] == "ZONE_END" for e in r)]
    ncore = len({(e[3], e[4]) for e in runs[-1]})
    print(f"SPAN grid={os.environ.get('I2M_POST_GRID', '9,4')} cores={ncore} runs={len(spans)} "
          f"median={statistics.median(spans[1:]):.2f} min={min(spans[1:]):.2f} us  all={[round(s, 1) for s in spans]}")


if __name__ == "__main__":
    main()
