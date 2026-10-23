"""Per-op device zone durations from a device-profiler profile_log_device.csv (device 0 by default):
for every op (run host ID), each zone's min / median / max duration over cores, in us.

    python zones.py <profile_log_device.csv> [chip]
"""
import collections
import csv
import statistics
import sys

path = sys.argv[1]
f = open(path)
freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0]) / 1e3  # cycles per ns
rows = csv.reader(f)
hdr = [h.strip() for h in next(rows)]
ix = {h: i for i, h in enumerate(hdr)}
start = {}
dur = collections.defaultdict(lambda: collections.defaultdict(list))
first = {}
for r in rows:
    r = [c.strip() for c in r]
    if len(sys.argv) > 2 and r[ix["PCIe slot"]] != sys.argv[2]:
        continue
    run = int(r[ix["run host ID"]] or 0)
    key = (run, r[ix["core_x"]], r[ix["core_y"]], r[ix["RISC processor type"]], r[ix["zone name"]])
    t = int(r[ix["time[cycles since reset]"]])
    first[run] = min(first.get(run, t), t)
    if r[ix["type"]] == "ZONE_START":
        start[key] = t
    elif r[ix["type"]] == "ZONE_END" and key in start:
        dur[run][(key[3], key[4])].append((t - start.pop(key)) / freq / 1e3)
for run in sorted(dur):
    print(f"== op run {run}")
    for (risc, zone), v in sorted(dur[run].items()):
        print(f"   {risc:7s} {zone[:28]:28s} n {len(v):3d}  min {min(v):6.2f}  med {statistics.median(v):6.2f}  max {max(v):6.2f} us")
