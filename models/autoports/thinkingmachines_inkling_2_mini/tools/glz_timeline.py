# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Event timeline of one glue program (OP=gate: tt/gate.py GateKernel2 like the MLP-side gate, NORM + ROUTER, T rows)
run N times on one die with the device profiler: every zone start and GLZ point marker (tt/kernels/glz.hpp), as us
after the run's first kernel start (the earliest core), averaged over runs 2..N; for the root core (the lowest x, y)
and for the median other core, per RISC, in time order. Shows where a cross-core round's 3-5 us go: the partial's
compute (C_PP), the send (W_S), the gather wait (W_G), the root reduce (C_TOUT / W_T), the broadcast (W_D).
env: N (10), T (1), OP (gate)"""

import collections
import csv
import os
import statistics

import torch

import ttnn
from models.autoports.thinkingmachines_inkling_2_mini.tt.gate import GateKernel2
from models.autoports.thinkingmachines_inkling_2_mini.tt.post import PostKernel


def run_gate(mesh, n, T):
    H, Rk = 2304, 64
    pk = PostKernel(mesh, H, 1e-6)
    k = GateKernel2(pk, Rk)
    L1 = ttnn.L1_MEMORY_CONFIG
    up = lambda t, dt, mc=None: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh,  # noqa: E731
                                                memory_config=mc)
    h, m = up(torch.randn(1, 1, T, H), ttnn.float32, L1), up(torch.randn(1, 1, T, H), ttnn.float32, L1)
    wd = up(torch.randn(1, 1, 2 * H, Rk) * 0.05, ttnn.bfloat16)
    wfg = up(torch.randn(1, 1, Rk, 2 * H) * 0.3, ttnn.bfloat16)
    nw = up(torch.ones(1, 1, 32, H), ttnn.float32)
    rt = (up(torch.randn(1, 1, H, 128) * 0.02, ttnn.bfloat16),) * 2
    for _ in range(n):
        k(h, m, wd, wfg, nw=nw, router=rt, dup=T == 1)


def run_post(mesh, n, T):
    from models.autoports.thinkingmachines_inkling_2_mini.tt.batched import BatchedConv

    H, NB = 2304, int(os.environ.get("NB", "3"))
    k = PostKernel(mesh, H, 1e-6)
    l1 = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh,  # noqa: E731
                                       memory_config=ttnn.L1_MEMORY_CONFIG)
    att, x = l1(torch.randn(1, 1, T, H), ttnn.bfloat16), l1(torch.randn(1, 1, T, H), ttnn.bfloat16)
    conv = BatchedConv(mesh, torch.randn(H, 4) * 0.3, None, H, T)
    q = ttnn.from_torch(torch.randn(1, 1, 32, H) * 0.02, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
    blocks = [l1(torch.randn(1, 1, T, H), ttnn.float32) for _ in range(NB)]
    lg = [(l1(torch.randn(1, 1, T, 64), ttnn.float32), 1) for _ in range(NB)]
    nw = k.rows_weight(ttnn.from_torch(torch.ones(1, 1, 1, H), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                       device=mesh))
    if os.environ.get("L1C") == "1":  # (probe: the per-layer constants and conv states L1-resident instead of DRAM)
        L1 = ttnn.L1_MEMORY_CONFIG
        conv.w = [ttnn.to_memory_config(t, L1) for t in conv.w]
        conv.w_cur = ttnn.to_memory_config(conv.w_cur, L1)
        conv.s = [ttnn.to_memory_config(t, L1) for t in conv.s]
        q = ttnn.to_memory_config(q, L1)
        nw = ttnn.to_memory_config(nw, L1)
    E = int(os.environ.get("E", "128"))
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)  # noqa: E731
    rt = (up(torch.randn(1, 1, H, E) * 0.02), up(torch.randn(1, 1, H, E) * 1e-4)) if os.environ.get("ROUTER", "1") == "1" \
        else None
    for _ in range(n):
        k(att, x, conv, q, blocks, lg, nw=nw, router=rt, dup=T == 1)


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    n, T = int(os.environ.get("N", "10")), int(os.environ.get("T", "1"))
    {"gate": run_gate, "post": run_post}[os.environ.get("OP", "gate")](mesh, n, T)
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
    ev = []  # (t, x, y, risc, name, type)
    for r in rows[2:]:
        ev.append((int(r[ix["time[cycles since reset]"]]), int(r[ix["core_x"]]), int(r[ix["core_y"]]),
                   r[ix["RISC processor type"]].strip(), r[ix["zone name"]].strip(), r[ix["type"]].strip()))
    ev.sort()
    # runs: split at each BRISC-KERNEL start of the root core
    root = min((x, y) for _, x, y, risc, nm, ty in ev if nm == "BRISC-KERNEL")
    starts = [t for t, x, y, risc, nm, ty in ev if (x, y) == root and nm == "BRISC-KERNEL" and ty == "ZONE_START"]
    starts.append(1 << 62)
    per = collections.defaultdict(list)  # (core, risc, name, k-th occurrence) -> [us after run start]
    for i in range(1, len(starts) - 1):  # (skip run 0: compile / first touch)
        lo, hi = starts[i] - 20000, starts[i + 1] - 20000
        seg = [e for e in ev if lo <= e[0] < hi and e[5] == "ZONE_START"]
        if not seg:
            continue
        t0 = min(e[0] for e in seg if e[4].endswith("-KERNEL"))
        cnt = collections.Counter()
        for t, x, y, risc, nm, ty in seg:
            key = ((x, y), risc, nm)
            per[key + (cnt[key],)].append((t - t0) / freq)
            cnt[key] += 1
    cores = sorted({k[0] for k in per})
    ends = {c: max(statistics.mean(v) for k, v in per.items() if k[0] == c) for c in cores}
    others = sorted((c for c in cores if c != root), key=lambda c: ends[c])
    med = others[len(others) // 2] if others else root
    allk = sorted({(k[1], k[2], k[3]) for k in per})
    tab = []
    for risc, nm, occ in allk:
        vals = {c: statistics.mean(per[(c, risc, nm, occ)]) for c in cores if (c, risc, nm, occ) in per}
        ov = [v for c, v in vals.items() if c != root]
        if not ov:
            continue
        slow = max(ov)
        tab.append((vals.get(root, float("nan")), statistics.median(ov), slow, max(vals, key=vals.get), risc, nm, occ))
    print("== event: root / median other / max other (core) us after the first kernel start")
    for r, md, mx, c, risc, nm, occ in sorted(tab, key=lambda x: x[1]):
        print(f"  {r:7.2f} {md:7.2f} {mx:7.2f} {str(c):9s} {risc:8s} {nm}#{occ}")
    print("== done")


if __name__ == "__main__":
    main()
