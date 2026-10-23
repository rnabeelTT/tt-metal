"""Per-step device breakdown of the traced decode windows (signposted) in an ops_perf_results CSV. Device 0 rows."""
import collections
import csv
import re
import sys

rows = list(csv.DictReader(open(sys.argv[1])))
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 5
names = sys.argv[3].split(",") if len(sys.argv) > 3 else ["gdn", "attn"]
wins, cur = [], None
for x in rows:
    if x["OP TYPE"] == "signpost":
        if x["OP CODE"].endswith("_END"):
            wins.append(cur)
            cur = None
        else:
            cur = []
    elif cur is not None and x["DEVICE ID"] == "0":
        cur.append(x)


def f(x, k):
    try:
        return float(x[k] or 0) / 1e3
    except ValueError:
        return 0.0


for name, w in zip(names, wins):
    n = len(w) / iters
    k = sum(f(x, "DEVICE KERNEL DURATION [ns]") for x in w) / iters
    o = sum(f(x, "OP TO OP LATENCY [ns]") for x in w) / iters
    print(f"== {name}: {n:.0f} ops/step, kernel {k:.1f} us, op-to-op gaps {o:.1f} us, total {k+o:.1f} us")
    agg = collections.defaultdict(lambda: [0, 0.0, 0.0])
    for x in w:
        a = agg[re.sub("DeviceOperation", "", x["OP CODE"])]
        a[0] += 1
        a[1] += f(x, "DEVICE KERNEL DURATION [ns]")
        a[2] += f(x, "OP TO OP LATENCY [ns]")
    for op, (c, kk, oo) in sorted(agg.items(), key=lambda t: -(t[1][1] + t[1][2])):
        print(f"   {op[:44]:44s} x{c/iters:4.0f}  kernel {kk/iters:7.1f} us  gaps {oo/iters:6.1f} us")
