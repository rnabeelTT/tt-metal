# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-7 perf tables: the reduced one-layer-per-kind Tracy windows with the stage-7 decode knobs (doc/optimized_full_model/
tracy/<kind>_<case>_mesh_perf_report.csv, per-device rows) against the stage-5 tables (doc/optimized_multichip_decoder/tracy/
<kind>/<case>_mesh_perf_report.csv), plus the 48-layer lower bound they imply.

    python models/autoports/qwen_qwen3_5_122b_a10b/tools/perf_tables_ofm.py > doc/optimized_full_model/PERF_TABLES.md
"""
import collections
import csv
import os
import re

HERE = os.path.dirname(__file__)
BEFORE = os.path.join(HERE, "..", "doc", "optimized_multichip_decoder", "tracy")
AFTER = os.path.join(HERE, "..", "doc", "optimized_full_model", "tracy")
KINDS = {"gdn": ("GDN (L0)", 36), "attn": ("attention (L3)", 12)}
CASES = (
    ("prefill_128", "prefill_128"),
    ("prefill_256", "prefill_256"),
    ("prefill_2048", "prefill_2048"),
    ("decode", "traced_decode"),
)


def rows(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else None


def by_device(rs):
    out = collections.defaultdict(list)
    for r in rs:
        out[r.get("Device", "0")].append(r)
    return dict(sorted(out.items()))


def replays(rs):
    """Traced-decode windows hold several replays: count them by the reduce-scatters (2 per layer per replay)."""
    n = sum(1 for r in rs if "ReduceScatter" in r["OP Code"])
    return max(1, n // 2)


def dev_ms(rs, div):
    return sum(float(r["Device Time"] or 0) for r in rs) / 1e3 / div


def gaps_ms(rs, div):
    return (
        sum(float(r["Op-to-Op Gap"] or 0) for r in rs if r["Op-to-Op Gap"] and float(r["Op-to-Op Gap"]) < 1000)
        / 1e3
        / div
    )


def main():
    print("# Stage-7 perf tables (Tracy, tt-perf-report signposted windows, reduced one-layer-per-kind model)\n")
    print(
        "Before = stage-5 `MULTICHIP_SELECTED` (committed tables, `doc/optimized_multichip_decoder/tracy`); after = the stage-7 selected\n"
        "policy (seven decode op-count knobs + sorted_min_tokens=256; prefill_256 has no stage-5 row). Device time = sum over the window per device (decode: per traced replay); ops = per replay.\n"
    )
    print(
        "| layer | case | before ms/device 0/1/2/3 (ops) | after ms/device 0/1/2/3 (ops) | after gaps ms (dev 0) | delta (max device) |"
    )
    print("|---|---|---|---|---|---|")
    bound_before = bound_after = 0.0
    for k, (kname, nlayers) in KINDS.items():
        for bcase, acase in CASES:
            b = rows(os.path.join(BEFORE, k, f"{bcase}_mesh_perf_report.csv"))
            a = rows(os.path.join(AFTER, f"{k}_{acase}_mesh_perf_report.csv"))

            def fmt(rs):
                if rs is None:
                    return "n/a", None, None
                bd = by_device(rs)
                div = replays(list(bd.values())[0]) if "decode" in (bcase + acase) else 1
                vals = [dev_ms(v, div) for v in bd.values()]
                ops = [len(v) // div for v in bd.values()]
                return (
                    "/".join(f"{v:.3f}" for v in vals) + f" ({'/'.join(str(o) for o in ops)})",
                    max(vals),
                    (list(bd.values())[0], div),
                )

            bs, bmax, _ = fmt(b)
            as_, amax, a0 = fmt(a)
            gap = f"{gaps_ms(a0[0], a0[1]):.3f}" if a0 else "n/a"
            delta = (
                f"{amax - bmax:+.3f} ms ({100 * (amax - bmax) / bmax:+.1f} %)" if (amax is not None and bmax) else "n/a"
            )
            print(f"| {kname} | {bcase} | {bs} | {as_} | {gap} | {delta} |")
            if bcase == "decode":
                if bmax is not None:
                    bound_before += nlayers * bmax
                if amax is not None:
                    bound_after += nlayers * amax
    print()
    print(
        f"48-layer decode layer-stack device lower bound (36 x GDN + 12 x attention, max over devices): before {bound_before:.2f} ms, "
        f"after {bound_after:.2f} ms."
    )
    # per-op table of the after decode windows
    for k, (kname, _) in KINDS.items():
        a = rows(os.path.join(AFTER, f"{k}_traced_decode_mesh_perf_report.csv"))
        if a is None:
            continue
        d0 = by_device(a)
        d0 = list(d0.values())[0]
        div = replays(d0)
        cnt = collections.Counter(re.sub(r"DeviceOperation", "", r["OP Code"])[:56] for r in d0)
        print(f"\n## {kname} traced decode, device 0, per replay ({len(d0) // div} ops)\n")
        print("| op | count | device us | gaps us |")
        print("|---|---|---|---|")
        for op, c in cnt.most_common():
            t = (
                sum(float(r["Device Time"] or 0) for r in d0 if re.sub(r"DeviceOperation", "", r["OP Code"])[:56] == op)
                / div
            )
            g = (
                sum(
                    float(r["Op-to-Op Gap"] or 0)
                    for r in d0
                    if re.sub(r"DeviceOperation", "", r["OP Code"])[:56] == op
                    and r["Op-to-Op Gap"]
                    and float(r["Op-to-Op Gap"]) < 1000
                )
                / div
            )
            print(f"| {op} | {c // div} | {t:.1f} | {g:.1f} |")


if __name__ == "__main__":
    main()
