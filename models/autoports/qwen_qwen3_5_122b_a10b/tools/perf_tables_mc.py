# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Render the stage-4 mesh Tracy tables (doc/multichip_decoder/tracy/<kind>/<case>_mesh_perf_report.csv, per-device rows from
`tt-perf-report --no-merge-devices`) against the stage-3 single-chip optimized tables (doc/optimized_decoder/tracy/<kind>_optimized).

    python models/autoports/qwen_qwen3_5_122b_a10b/tools/perf_tables_mc.py > doc/multichip_decoder/PERF_TABLES.md
"""
import collections
import csv
import os
import re

HERE = os.path.dirname(__file__)
MESH = os.path.join(HERE, "..", "doc", "multichip_decoder", "tracy")
SINGLE = os.path.join(HERE, "..", "doc", "optimized_decoder", "tracy")
KINDS = {"gdn": "GDN (L0)", "attn": "attention (L3)"}
CASES = (("prefill_128", 1), ("prefill_2048", 1), ("decode", 3))  # decode window = 3 traced replays
TP = 4
CCL = ("AllGather", "ReduceScatter", "AllReduce")


def rows(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else None


def ms(rs, div=1):
    return sum(float(r["Device Time"] or 0) for r in rs) / 1e3 / div


def by_device(rs):
    out = collections.defaultdict(list)
    for r in rs:
        out[r.get("Device", "0")].append(r)
    return dict(sorted(out.items()))


def short(op):
    op = re.sub(r"DeviceOperation", "", op)
    return op[:60]


def main():
    print("# Stage-4 mesh perf tables (Tracy, tt-perf-report signposted windows)\n")
    print(
        "Single-chip = stage-3 `OptimizedDecoder`/`SELECTED` on one die (committed tables, `doc/optimized_decoder/tracy/*_optimized`)."
    )
    print(
        "Mesh = `MultichipDecoder`/`MULTICHIP_SELECTED` on the (1,4) P300x2, device time summed over the window on each device"
    )
    print(
        "(decode = one traced replay, 3 collected). Speedup = single-chip / max over devices; efficiency = speedup / 4.\n"
    )
    print(
        "| layer | case | single-chip ms (ops) | mesh ms per device 0/1/2/3 (ops/device) | speedup | efficiency | CCL ms/device |"
    )
    print("|---|---|---|---|---|---|---|")
    tops = {}
    for k, kname in KINDS.items():
        for case, div in CASES:
            s = rows(os.path.join(SINGLE, f"{k}_optimized", f"{case}_perf_report.csv"))
            m = rows(os.path.join(MESH, k, f"{case}_mesh_perf_report.csv"))
            if not m:
                print(f"| {kname} | {case} | n/a | n/a | | | |")
                continue
            dev = by_device(m)
            per = {d: ms(rs, div) for d, rs in dev.items()}
            worst = max(per.values())
            ccl = {d: ms([r for r in rs if any(c in r["OP Code"] for c in CCL)], div) for d, rs in dev.items()}
            nops = len(next(iter(dev.values()))) // div
            s_ms = ms(s, div) if s else float("nan")
            sp = s_ms / worst if s else float("nan")
            print(
                f"| {kname} | {case} | {s_ms:.3f} ({len(s)//div if s else 0}) | "
                + "/".join(f"{per[d]:.3f}" for d in per)
                + f" ({nops}) | {sp:.2f}x | {100*sp/TP:.0f}% | "
                + "/".join(f"{ccl[d]:.3f}" for d in ccl)
                + " |"
            )
            tops[(k, case)] = (dev, div)
    for (k, case), (dev, div) in tops.items():
        d0 = dev["0"]
        agg = collections.defaultdict(lambda: [0.0, 0])
        for r in d0:
            key = (short(r["OP Code"]), r.get("Math Fidelity", ""), r.get("Cores", ""))
            agg[key][0] += float(r["Device Time"] or 0) / 1e3 / div
            agg[key][1] += 1
        total = ms(d0, div)
        top = sorted(agg.items(), key=lambda kv: -kv[1][0])[:10]
        print(f"\n### {KINDS[k]} {case} — device 0 top ops ({total:.3f} ms, {len(d0)//div} ops per window)\n")
        print("| op (shape) | fidelity | cores | count | ms | share |")
        print("|---|---|---|---|---|---|")
        for (op, fid, cores), (t, n) in top:
            print(f"| {op} | {fid} | {cores} | {n//div if div>1 else n} | {t:.3f} | {100*t/total:.0f}% |")
        gaps = sum(float(r["Op-to-Op Gap"] or 0) for r in d0) / 1e3 / div
        print(f"\nOp-to-op gaps on device 0: {gaps:.3f} ms per window.")


if __name__ == "__main__":
    main()
