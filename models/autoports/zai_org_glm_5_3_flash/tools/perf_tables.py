# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarize the Tracy / tt-perf-report windows of a stage: one row per case (ops, device time, op-to-op gaps, host wall)
and the top op groups per case. Usage: python -m ...tools.perf_tables <stage doc dir> [--top N]"""

import argparse
import collections
import csv
import re
from pathlib import Path


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("doc_dir")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()
    root = Path(args.doc_dir) / "tracy"
    print("| case | ops / pass | device time (us) | op-to-op gaps (us) | host wall |")
    print("|---|---|---|---|---|")
    tops = {}
    for d in sorted(root.iterdir()):
        rep = d / "perf_report.csv"
        if not rep.exists():
            continue
        rows = [r for r in csv.DictReader(open(rep)) if r.get("OP Code") and r.get("Device Time")]
        reps = 3 if "decode" in d.name else 1
        step = rows[: len(rows) // reps]
        dev = sum(f(r["Device Time"]) for r in step)
        gap = sum(f(r["Op-to-Op Gap"]) for r in step)
        wall = ""
        con = d / "console.log"
        if con.exists():
            m = re.findall(r"^PERF .*?: ([0-9.]+ ms[^ ]*)", con.read_text(errors="ignore"), re.M)
            wall = m[-1] if m else ""
        print(f"| {d.name.strip('_')} | {len(step)} | {dev:.0f} | {gap:.0f} | {wall} |")
        agg = collections.defaultdict(lambda: [0, 0.0])
        for r in step:
            a = agg[r["OP Code"]]
            a[0] += 1
            a[1] += f(r["Device Time"])
        tops[d.name] = sorted(agg.items(), key=lambda kv: -kv[1][1])[: args.top]
    for name, top in tops.items():
        print(f"\n{name.strip('_')}: " + "; ".join(f"{k} x{c} {t:.0f} us" for k, (c, t) in top))


if __name__ == "__main__":
    main()
