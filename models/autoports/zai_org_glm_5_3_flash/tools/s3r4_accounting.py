# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-3 review #4: per-op-group device time (µs per layer) from tt-perf-report CSVs (perf_report.csv of a Tracy case
directory), and an A/B per op code between two case directories.

  groups DIR [DIR ...]          one column per case; decode = one traced replay (report sum / 3), prefill = one pass
  diff   DIR_A DIR_B [--min 2]  per op code (with the input-0 shape from ops.csv[.gz]): B - A, µs per replay

usage: python -m models.autoports.zai_org_glm_5_3_flash.tools.s3r4_accounting groups <dir> ...
       python -m models.autoports.zai_org_glm_5_3_flash.tools.s3r4_accounting diff <dir_a> <dir_b>"""

import collections
import csv
import gzip
import os
import re
import sys

GROUPS = [  # (group, op-code regex); the first match wins
    ("Routed sparse matmuls", r"^SparseMatmul"),
    ("Dense matmuls", r"^Matmul"),
    ("Norms", r"^LayerNorm"),
    ("Sinkhorn", r"^MhcSplitSinkhorn"),
    ("mHC pre", r"^MhcPre"),
    ("mHC post", r"^MhcPost"),
    ("mHC mix", r"^MhcMix"),
    ("KDA conv op", r"^KdaDecodeConv"),
    ("KDA recurrence", r"^KdaRecurrent|^Kda"),
    ("Router TopK", r"^TopKDevice"),
    (
        "DSA attention (sparse SDPA, indexer scores / top-k, cache writes)",
        r"SDPA|^DsaDecode|^Topk|^PagedUpdate|^Scatter",
    ),
    ("MoE glue (dispatch, gather rows)", r"^Moe"),
    (
        "Data movement",
        r"^Reshape|^Transpose|^Permute|^Slice|^Concat|^Tilize|^Untilize|Sharded|^Copy|^FillPad|^Pad|^Repeat",
    ),
    ("Element-wise / other", r"."),
]


def _reps(d):
    return 1 if "prefill" in os.path.basename(d.rstrip("/")) else 3


def _rows(d):
    with open(os.path.join(d, "perf_report.csv")) as f:
        return list(csv.DictReader(f))


def _op(r):
    return r["OP Code"].split(" ")[0]


def groups(dirs):
    cols = []
    for d in dirs:
        g = collections.OrderedDict((n, 0.0) for n, _ in GROUPS)
        n = _reps(d)
        for r in _rows(d):
            t = float(r["Device Time"] or 0) / n
            for name, pat in GROUPS:
                if re.search(pat, _op(r)):
                    g[name] += t
                    break
        cols.append(g)
    names = [os.path.basename(d.rstrip("/")) for d in dirs]
    print("| Group | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(dirs))
    for name, _ in GROUPS:
        print(f"| {name} | " + " | ".join(f"{c[name]:.0f}" for c in cols) + " |")
    print("| **Total** | " + " | ".join(f"**{sum(c.values()):.0f}**" for c in cols) + " |")


def _ops_csv(d):
    for n in ("ops.csv.gz", "ops.csv"):
        p = os.path.join(d, n)
        if os.path.exists(p):
            return p
    return None


def _shape_rows(d, sp):
    """Signposted ops.csv rows: (op code, input-0 shape) -> total device kernel µs."""
    p = _ops_csv(d)
    opener = gzip.open if p.endswith(".gz") else open
    on, agg = False, collections.defaultdict(float)
    with opener(p, "rt") as f:
        for x in csv.DictReader(f):
            if x["OP TYPE"] == "signpost":
                on = x["OP CODE"] == sp if x["OP CODE"] in (sp, sp + "_END") else on
                continue
            if on:
                shp = "x".join(x.get(f"INPUT_0_{a}_PAD[LOGICAL]", "") for a in "WZYX")
                agg[(x["OP CODE"], shp)] += float(x["DEVICE KERNEL DURATION [ns]"] or 0) / 1000
    return agg


def diff(a, b, minimum=2.0):
    sp = "PERF_PREFILL" if "prefill" in a else "PERF_DECODE"
    n = _reps(a)
    A, B = _shape_rows(a, sp), _shape_rows(b, sp)
    keys = set(A) | set(B)
    tot = (sum(B.values()) - sum(A.values())) / n
    print(f"total {sum(A.values()) / n:.1f} -> {sum(B.values()) / n:.1f} us ({tot:+.1f})")
    for k in sorted(keys, key=lambda k: (B.get(k, 0) - A.get(k, 0))):
        dv = (B.get(k, 0) - A.get(k, 0)) / n
        if abs(dv) >= minimum:
            print(f"{dv:+9.1f} us  {A.get(k, 0) / n:8.1f} -> {B.get(k, 0) / n:8.1f}  {k[0]} {k[1]}")


if __name__ == "__main__":
    if sys.argv[1] == "groups":
        groups(sys.argv[2:])
    else:
        diff(sys.argv[2], sys.argv[3], float(sys.argv[5]) if len(sys.argv) > 5 else 2.0)
