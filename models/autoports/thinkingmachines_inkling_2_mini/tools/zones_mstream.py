# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device zones of the streamed q | r | s | g, k, v call (tt/mstream.py) at TP's per-die shard shapes on one die: each
VARIANTS entry (tools/bench_mstream.py syntax) run N times eagerly; per run the program span (first kernel start to last
end over all cores), then per RISC the kernel start / end offsets from the span start, mean over runs, per core group
(q readers, k | v readers ..). env: TP, VARIANTS, N (5), R (1); the profiler CSV under TT_METAL_PROFILER_DIR."""

import collections
import csv
import glob
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(__file__))
import bench_mstream as B  # noqa: E402


def main():
    TP, R, N = int(os.environ.get("TP", "4")), int(os.environ.get("R", "1")), int(os.environ.get("N", "5"))
    variants = B.parse(os.environ["VARIANTS"])
    assert len(variants) == 1, "one variant per run (zones are not tagged per program)"
    dev, K, fqp, kd, ws = B.setup(TP)
    name, kv = variants[0]
    ms = B.make(dev, K, fqp, kd, kv)
    xs = [ttnn.from_torch(torch.randn(1, 1, R, K).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                          device=dev, memory_config=ttnn.L1_MEMORY_CONFIG) for _ in range(3)]
    for _ in range(N + 1):
        ms(xs, ws, R)
        ttnn.synchronize_device(dev)
    if os.environ.get("TRACE") == "1":  # (then N calls back to back in one trace: the gaps between programs)
        ttnn.ReadDeviceProfiler(dev)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        for _ in range(N):
            ms(xs, ws, R)
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        ttnn.release_trace(dev, tid)
    ttnn.ReadDeviceProfiler(dev)
    groups = {}
    for gi, (dl, _, g, _) in enumerate(ms.parts):
        for c, b, c0, nc in dl.assign:
            w = dev.worker_core_from_logical_core(c)
            groups[(w.x, w.y)] = (gi, b, nc)
    ttnn.close_mesh_device(dev)
    path = glob.glob(os.environ["TT_METAL_PROFILER_DIR"] + "/**/profile_log_device.csv", recursive=True,
                     include_hidden=True)[0]
    rows = list(csv.reader(open(path)))
    freq = float(",".join(rows[0]).split("CHIP_FREQ[MHz]:")[1].split(",")[0].strip())
    hdr = [c.strip() for c in rows[1]]
    ix = {n: i for i, n in enumerate(hdr)}
    ev = collections.defaultdict(list)  # (x, y, risc, zone) -> [(start, end)]
    open_ = {}
    for r in rows[2:]:
        zn = r[ix["zone name"]].strip()
        if not (zn.endswith("-KERNEL") or zn.endswith("-FW") or zn.startswith("MS_")):
            continue
        key = (int(r[ix["core_x"]]), int(r[ix["core_y"]]), r[ix["RISC processor type"]].strip(), zn)
        t = int(r[ix["time[cycles since reset]"]])
        if r[ix["type"]].strip() == "ZONE_START":
            open_[key] = t
        elif key in open_:
            ev[key].append((open_.pop(key), t))
    # runs: every core's i-th kernel zone is run i (the first, a compile / warm-up run, dropped)
    starts = collections.defaultdict(list)
    for (x, y, risc, zn), se in ev.items():
        if (x, y) in groups and zn.endswith("-KERNEL"):
            for i, (s, e) in enumerate(se):
                starts[i].append(s)
    t0 = {i: min(v) for i, v in starts.items()}
    span = collections.defaultdict(lambda: [1e30, 0])
    agg = collections.defaultdict(list)  # (group, risc, zone) -> [(start off, end off)] over cores / runs
    for (x, y, risc, zn), se in ev.items():
        if (x, y) not in groups:
            continue
        gi = groups[(x, y)][0]
        if zn.endswith("-FW"):
            continue
        for i, (s, e) in enumerate(se[1:], 1):
            agg[(gi, risc, zn)].append(((s - t0[i]) / freq, (e - t0[i]) / freq))
            span[i][0], span[i][1] = min(span[i][0], s), max(span[i][1], e)
    fw = collections.defaultdict(lambda: [1e30, 0])
    for (x, y, risc, zn), se in ev.items():
        if (x, y) in groups and zn.endswith("-FW"):
            for i, (s, e) in enumerate(se[1:], 1):
                fw[i][0], fw[i][1] = min(fw[i][0], s), max(fw[i][1], e)
    fws = [(e - s) / freq for i, (s, e) in fw.items()]
    rel = collections.defaultdict(list)  # (group, risc, zone): start / end offsets from the FW span start
    for (x, y, risc, zn), se in ev.items():
        if (x, y) in groups:
            for i, (s, e) in enumerate(se[1:], 1):
                rel[(groups[(x, y)][0], risc, zn)].append(((s - fw[i][0]) / freq, (e - fw[i][0]) / freq))
    for k, v in sorted(rel.items()):
        if k[1] in ("BRISC", "NCRISC", "TRISC_1"):
            print(f"  fw-rel group {k[0]} {k[1]:7s} {k[2]:14s} start min {min(a for a, _ in v):6.2f} mean "
                  f"{sum(a for a, _ in v) / len(v):6.2f} max {max(a for a, _ in v):6.2f}  end mean "
                  f"{sum(b for _, b in v) / len(v):6.2f} max {max(b for _, b in v):6.2f}")
    print(f"{name} R{R}: FW span mean {sum(fws) / len(fws):.2f} us (runs {', '.join(f'{v:.2f}' for v in fws)})")
    sp = [(e - s) / freq for i, (s, e) in span.items() if i > 0]
    ks = sorted((s, e) for i, (s, e) in span.items())
    if os.environ.get("TRACE") == "1":
        print("  traced: kernel spans", [round((e - s) / freq, 2) for s, e in ks[-N:]], "gaps to the next start",
              [round((ks[j + 1][0] - ks[j][1]) / freq, 2) for j in range(len(ks) - N, len(ks) - 1)],
              "start to start", [round((ks[j + 1][0] - ks[j][0]) / freq, 2) for j in range(len(ks) - N, len(ks) - 1)])
    print(f"{name} R{R}: program span mean {sum(sp) / len(sp):.2f} us (runs {', '.join(f'{v:.2f}' for v in sp)})")
    for (gi, risc, zn), v in sorted(agg.items()):
        so, eo = [a for a, _ in v], [b for _, b in v]
        print(f"  group {gi} {risc:8s} {zn:14s} start mean {sum(so) / len(so):6.2f} max {max(so):6.2f}  end mean "
              f"{sum(eo) / len(eo):6.2f} min {min(eo):6.2f} max {max(eo):6.2f} us  ({len(v)})", flush=True)
    # per core, the reader end (the slowest cores)
    late = sorted(((sum(e for _, e in se[1:]) - sum(t0[i] for i in range(1, len(se)))) / (len(se) - 1) / freq, x, y,
                   groups[(x, y)]) for (x, y, risc, zn), se in ev.items() if (x, y) in groups and risc == "NCRISC"
                  and zn.endswith("-KERNEL"))
    print("  latest readers (end us, core, (group, bank, cols)):", [(round(a, 2), (x, y), g) for a, x, y, g in late[-8:]])
    print("  earliest readers:", [(round(a, 2), (x, y), g) for a, x, y, g in late[:4]])


if __name__ == "__main__":
    main()
