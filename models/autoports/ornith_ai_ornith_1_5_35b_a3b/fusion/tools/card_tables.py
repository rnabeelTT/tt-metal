"""Model-card performance tables from card_bench.sh runs: published vs new side by side.

    python card_tables.py <new label prefix> <published label prefix> <profile> [...]   (labels: <prefix>_<profile>)
"""

import json
import os
import re
import sys
from pathlib import Path

RUNS = Path(os.environ.get("RUNS", Path.home() / "ornith-fusion" / "runs"))
# Profiles whose decode code is unchanged (e.g. one die falls back to the v2 MoE): their single-user random-token rows
# measure only adaptive-MTP run-to-run variance, so the latency table is replaced by SAME_DECODE_NOTE.
SAME_DECODE = set(filter(None, os.environ.get("SAME_DECODE", "").split(",")))
SAME_DECODE_NOTE = os.environ.get("SAME_DECODE_NOTE", "Decode code unchanged on this profile; single-user rows omitted.")


def load(label):
    out = {}
    for f in (RUNS / label).glob("bench_*.json"):
        isl, osl, u = map(int, re.match(r"bench_(\d+)_(\d+)_u(\d+)\.json", f.name).groups())
        d = json.loads(f.read_text())
        out[(isl, osl, u)] = {
            "n": d["completed"],
            "ttft": d["median_ttft_ms"],
            "tpot": d["median_tpot_ms"],
            "e2el": d["median_e2el_ms"] / 1e3,
            "out": d["output_throughput"],
            "prefill": isl / (d["median_ttft_ms"] / 1e3),
            "decode": 1e3 / d["median_tpot_ms"],
        }
    rt = RUNS / label / "realtext.txt"
    real = None
    if rt.exists():
        m = re.search(r"median ([\d.]+)", rt.read_text())  # per-prompt median: one near-fully-accepted prompt skews the total
        real = (float(m.group(1)),) if m else None
    return out, real


def pct(new, old, higher_better=True):
    if not old:
        return ""
    d = (new / old - 1) * 100
    return f"{d:+.0f} %"


def fmt(v, nd=1):
    return f"{v:,.{nd}f}"


def section(profile, new, pub):
    (rn, real_n), (rp, real_p) = new, pub
    lines = [f"#### {profile}", ""]
    lat = sorted(k for k in rn if k[2] == 1)
    if profile in SAME_DECODE:
        lat = []
        lines += [SAME_DECODE_NOTE]
    else:
        lines += [
            "Latency, batch 1 (single user): published v2 -> blackhole-fast",
            "",
            "| Input tokens | Output tokens | TTFT (ms) | Prefill tok/s | TPOT (ms) | Decode tok/s | E2EL (s) |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    for k in lat:
        a, b = rp.get(k), rn[k]
        if a is None:
            continue
        lines.append(
            f"| {k[0]:,} | {k[1]} | {fmt(a['ttft'], 0)} -> {fmt(b['ttft'], 0)} | {fmt(a['prefill'], 0)} -> {fmt(b['prefill'], 0)} "
            f"| {a['tpot']:.1f} -> {b['tpot']:.1f} | {a['decode']:.1f} -> **{b['decode']:.1f}** ({pct(b['decode'], a['decode'])}) "
            f"| {a['e2el']:.2f} -> {b['e2el']:.2f} |"
        )
    if real_n and real_p:
        lines += ["", f"Real text, single user (12 chat prompts x 384 output tokens, temperature 0, MTP on; median per prompt): "
                      f"{real_p[0]:.1f} -> **{real_n[0]:.1f} tok/s** ({pct(real_n[0], real_p[0])})."]
    for isl, osl in ((128, 128), (1024, 256)):
        conc = sorted(k for k in rn if k[0] == isl and k[1] == osl and k[2] > 1)
        if not conc:
            continue
        lines += [
            "",
            f"Concurrency, {isl} input / {osl} output tokens: published v2 -> blackhole-fast",
            "",
            "| Users | N | TTFT (ms) | TPOT (ms) | Decode tok/s/user | Total out tok/s | E2EL (s) |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for k in conc:
            a, b = rp.get(k), rn[k]
            if a is None:
                continue
            lines.append(
                f"| {k[2]} | {b['n']} | {fmt(a['ttft'], 0)} -> {fmt(b['ttft'], 0)} | {a['tpot']:.1f} -> {b['tpot']:.1f} "
                f"| {a['decode']:.1f} -> {b['decode']:.1f} | {a['out']:.1f} -> **{b['out']:.1f}** ({pct(b['out'], a['out'])}) "
                f"| {a['e2el']:.2f} -> {b['e2el']:.2f} |"
            )
    return lines


def summary(profiles, data):
    lines = [
        "| Profile | ISL | OSL | Concurrency | N | TTFT (ms) | Prefill tok/s/u | Decode tok/s/u | E2EL (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for p in profiles:
        rn = data[p][0][0]
        for k in ((128, 128, 1), (1024, 256, 32)):
            if k in rn and not (p in SAME_DECODE and k[2] == 1):
                b = rn[k]
                lines.append(f"| {p} | {k[0]} | {k[1]} | {k[2]} | {b['n']} | {fmt(b['ttft'], 0)} | {fmt(b['prefill'], 0)} "
                             f"| {b['decode']:.1f} | {b['e2el']:.2f} |")
    return lines


def main():
    new_p, pub_p, *profiles = sys.argv[1:]
    data = {p: (load(f"{new_p}_{p}"), load(f"{pub_p}_{p}")) for p in profiles}
    out = summary(profiles, data) + [""]
    for p in profiles:
        out += section(p, *data[p]) + [""]
    print("\n".join(out))


if __name__ == "__main__":
    main()
