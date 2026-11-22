"""Card performance tables from quest_chain.sh runs (runs/quest_<profile>/bench_*.json, realtext.txt).

    python card_perf.py p300x2 p300 p150 > perf.md
"""

import json
import os
import re
import sys
from pathlib import Path

RUNS = Path(os.environ.get("RUNS", Path.home() / "ornith-fusion" / "runs"))
DIES = {"p300x2": "four dies", "p300": "two dies", "p150": "one die"}


def table(profile):
    d = RUNS / f"quest_{profile}"
    rows = []
    for f in d.glob("bench_*.json"):
        isl, osl, u = map(int, re.match(r"bench_(\d+)_(\d+)_u(\d+)\.json", f.name).groups())
        b = json.loads(f.read_text())
        rows.append((u, isl, osl, b["median_ttft_ms"], b["median_tpot_ms"], b["output_throughput"]))
    out = [f"#### {profile} ({DIES.get(profile, profile)})", ""]
    rt = d / "realtext.txt"
    m = re.search(r"median ([\d.]+)", rt.read_text()) if rt.exists() else None
    if m:
        out += [f"Single user on real text (12 chat prompts x 384 output tokens, temperature 0): {float(m.group(1)):.0f} tok/s.", ""]
    out += ["| users | input / output tokens | TTFT (ms) | TPOT (ms) | tok/s per user | output tok/s, all users |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for u, isl, osl, ttft, tpot, thr in sorted(rows):
        out.append(f"| {u} | {isl:,} / {osl} | {ttft:,.0f} | {tpot:.1f} | {1e3 / tpot:.0f} | {thr:,.0f} |")
    return "\n".join(out) + "\n"


def main():
    print("### Performance\n")
    print("Served as shipped (`tt-model serve --profile <profile>`), vLLM's random-token benchmark with `ignore_eos`, "
          "medians; one request per user at a time.\n")
    for p in sys.argv[1:] or ["p300x2", "p300", "p150"]:
        print(table(p))


if __name__ == "__main__":
    main()
