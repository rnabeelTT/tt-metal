# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device time per op of one DSpark drafter step (tt/drafter.py: ctrl + commit of 4 rows + draft d), keyed by call site
(the op_prof.py mechanism: each top-level ttnn op followed by a sync and a profiler read; die 0, eager).

usage: ~/inkling2-work/jobs/die.sh 3 dprof "-e TT_METAL_DEVICE_PROFILER=1 -e TT_METAL_PROFILER_MID_RUN_DUMP=1
       -e TT_METAL_PROFILER_CPP_POST_PROCESS=1" python -u models/autoports/thinkingmachines_inkling_2_mini/tools/drafter_prof.py
env: DRAFT_D (3), SEQ=1 (every op in order)
"""

import collections
import os

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tools import op_prof as OP
from models.autoports.thinkingmachines_inkling_2_mini.tt import drafter as drafter_mod

FIXTURE = os.environ.get("FIXTURE", "/cache/fixtures/dspark_taps.pt")


def main():
    d = int(os.environ.get("DRAFT_D", "3"))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    OP.REC["mesh"] = mesh
    fx = torch.load(FIXTURE)
    r = next(iter(fx["texts"].values()))
    dr = drafter_mod.DSparkDrafter(mesh, None, max_len=r["taps"].shape[0] + 64, d_max=d)
    p = int(r["anchors"][-1]) + 1
    dr.commit(r["taps"][: p - 4], 0)
    x = torch.zeros(1, 4, 4, dr.H, dtype=torch.bfloat16)
    x[0] = r["taps"][p - 4:p].transpose(0, 1)
    tb = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
    dr.set_inputs(int(r["ids"][p]), p - 4, 4)
    for _ in range(2):
        dr._traced_body(tb, d, True)
    OP._read()
    OP.REC["on"] = True
    dr._traced_body(tb, d, True)
    OP.REC["on"] = False
    rows = OP.REC["rows"]
    print(f"== drafter step d={d}: {len(rows)} ops, {sum(x[2] for x in rows)} programs, device total "
          f"{sum(x[3] for x in rows) / 1e3:.3f} ms")
    comp = collections.defaultdict(lambda: [0.0, 0])
    site_t = collections.defaultdict(lambda: [0.0, 0])
    for name, site, n, us in rows:
        fn = site.split(" ")[-1]
        comp[fn][0] += us
        comp[fn][1] += 1
        s = site_t[(site, name)]
        s[0] += us
        s[1] += 1
    if os.environ.get("SEQ"):
        for i, (name, site, n, us) in enumerate(rows):
            print(f"{i:4d} {n:2d} {us:8.1f}  {name:32s} {site}")
    print("-- by function: us, ops")
    for k, (us, c) in sorted(comp.items(), key=lambda kv: -kv[1][0]):
        print(f"{us:9.1f} us {c:5d}  {k}")
    print("-- top call sites: us, calls, op")
    for (site, name), (us, c) in sorted(site_t.items(), key=lambda kv: -kv[1][0])[:45]:
        print(f"{us:9.1f} us {c:4d}  {name:34s} {site}")
    ttnn.close_mesh_device(mesh)
    print("== done")


if __name__ == "__main__":
    main()
