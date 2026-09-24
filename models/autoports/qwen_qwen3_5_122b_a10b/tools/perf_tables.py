# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Render the stage-3 Tracy before/after tables (doc/optimized_decoder/tracy/<kind>_<mode>/*_perf_report.csv) as markdown,
with the dominant decode rows (dtype / fidelity proof columns) and the DRAM roofline accounting.

    python models/autoports/qwen_qwen3_5_122b_a10b/tools/perf_tables.py
"""
import csv
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..", "doc", "optimized_decoder", "tracy")
KINDS = {"gdn": "GDN (L0)", "attn": "attention (L3)"}
MODES = ("fused", "optimized")
CASES = (("prefill_128", 1), ("prefill_2048", 1), ("decode", 3))  # decode window = 3 traced replays
DRAM_GBPS = 512.0  # Blackhole p150/p300 die peak (tt-perf-report constant)
BFP8, BFP4, BF16 = 1.0625, 0.5625, 2.0  # bytes per element incl. shared exponents


def load(kind, mode, case):
    p = os.path.join(ROOT, f"{kind}_{mode}", f"{case}_perf_report.csv")
    if not os.path.exists(p):
        return None
    return list(csv.DictReader(open(p)))


def dev_ms(rows, div):
    return sum(float(r["Device Time"] or 0) for r in rows) / 1e3 / div


def roofline_mb(kind):
    H, E_ACT, N, S = 3072, 8, 1024, 1024
    experts = E_ACT * (H * 2 * N + N * H) * BFP4
    shared = (H * (2 * S + 1) + S * H) * BFP8
    router = H * 256 * BF16
    if kind == "gdn":
        mix = (H * 20608 + 8192 * H) * BFP8
        state = 2 * 64 * 128 * 128 * 2  # read + write bf16
        extra = state
    else:
        mix = (H * 17408 + 8192 * H) * BFP8
        extra = 128 * 2 * 2 * 256 * 2  # KV read at 128 keys (2 KV heads, K+V, bf16)
    return (experts + shared + router + mix + extra) / 1e6


def main():
    print("### Device time (tt-perf-report signposted windows; decode = one traced replay, 3 collected)\n")
    print("| layer | case | fused ms | fused ops | optimized ms | optimized ops | change |")
    print("|---|---|---|---|---|---|---|")
    for kind, label in KINDS.items():
        for case, div in CASES:
            a, b = load(kind, "fused", case), load(kind, "optimized", case)
            if not a or not b:
                continue
            ma, mb = dev_ms(a, div), dev_ms(b, div)
            print(
                f"| {label} | {case} | {ma:.3f} | {len(a)//div} | {mb:.3f} | {len(b)//div} | {100*(mb-ma)/ma:+.1f}% |"
            )
    print("\n### Roofline accounting (decode, one die, bytes per token at the stored dtypes / 512 GB/s)\n")
    print(
        "| layer | bytes/token (MB) | roofline ms | device ms (optimized replay) | fraction | host wall ms/step (3 replays) |"
    )
    print("|---|---|---|---|---|---|")
    for kind, label in KINDS.items():
        b = load(kind, "optimized", "decode")
        if not b:
            continue
        mb = roofline_mb(kind)
        rl = mb / (DRAM_GBPS * 1e3) * 1e3
        dm = dev_ms(b, 3)
        wall = ""
        log = os.path.join(ROOT, f"{kind}_optimized", "decode_tracy.console.log")
        if os.path.exists(log):
            for line in open(log):
                if "ms/step" in line:
                    wall = line.strip().split("traced decode")[-1].split("ms/step")[0].strip()
        print(f"| {label} | {mb:.1f} | {rl:.3f} | {dm:.3f} | {100*rl/dm:.0f}% | {wall} |")
    print("\n### Dominant decode rows (optimized, one replay; dtype / fidelity as measured)\n")
    for kind, label in KINDS.items():
        b = load(kind, "optimized", "decode")
        if not b:
            continue
        n = len(b) // 3
        rows = sorted(b[:n], key=lambda r: -float(r["Device Time"] or 0))[:12]
        print(f"**{label}**\n")
        print("| op | us | cores | bound | DRAM % | fidelity | in0 dtype | weight dtype | in0_block_w |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            print(
                f"| {r['OP Code'][:56]} | {float(r['Device Time']):.1f} | {r['Cores']} | {r['Bound'][:8]} | {r['DRAM %'][:5]} | "
                f"{r['Math Fidelity'][:5]} | {r['Input 0 Datatype'].replace('BFLOAT','bf').replace('_B','')} | "
                f"{r['Input 1 Datatype'].replace('BFLOAT','bf').replace('_B','')} | {r['Inner Dim Block Size']} |"
            )
        print()
    print("### Op-class totals per decode replay (fused -> optimized, us)\n")
    for kind, label in KINDS.items():
        a, b = load(kind, "fused", "decode"), load(kind, "optimized", "decode")
        if not a or not b:
            continue

        def agg(rows):
            d = {}
            for r in rows:
                k = r["OP Code"].split(" ")[0]
                d[k] = d.get(k, 0) + float(r["Device Time"] or 0) / 3
            return d

        da, db = agg(a), agg(b)
        print(f"**{label}**\n")
        print("| op class | fused us | optimized us |")
        print("|---|---|---|")
        for k in sorted(set(da) | set(db), key=lambda k: -max(da.get(k, 0), db.get(k, 0)))[:14]:
            print(f"| {k} | {da.get(k,0):.1f} | {db.get(k,0):.1f} |")
        print()


if __name__ == "__main__":
    sys.exit(main())
