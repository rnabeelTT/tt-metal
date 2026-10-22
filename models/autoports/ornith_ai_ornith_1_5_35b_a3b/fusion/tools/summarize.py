"""Side-by-side table of `vllm bench serve` rows: python summarize.py <labelA> <labelB> [...]"""

import json
import os
import re
import sys
from pathlib import Path

RUNS = Path(os.environ.get("RUNS", Path.home() / "ornith-fusion" / "runs"))


def rows(label):
    out = {}
    for f in (RUNS / label).glob("bench_*.json"):
        isl, osl, u = map(int, re.match(r"bench_(\d+)_(\d+)_u(\d+)\.json", f.name).groups())
        d = json.loads(f.read_text())
        out[(u, isl, osl)] = d
    return out


labels = sys.argv[1:]
data = {l: rows(l) for l in labels}
keys = sorted(set().union(*[d.keys() for d in data.values()]))
hdr = "users  isl/osl     " + "".join(f"| {l[:22]:>22s} TTFT ms  TPOT ms  out tok/s " for l in labels)
print(hdr)
for k in keys:
    u, isl, osl = k
    line = f"{u:5d}  {isl:>6d}/{osl:<4d}"
    for l in labels:
        d = data[l].get(k)
        if d is None:
            line += f"| {'':>22s} {'-':>7s}  {'-':>7s}  {'-':>9s} "
        else:
            line += (
                f"| {'':>22s} {d['median_ttft_ms']:7.0f}  {d['median_tpot_ms']:7.2f}  {d['output_throughput']:9.1f} "
            )
    print(line)
