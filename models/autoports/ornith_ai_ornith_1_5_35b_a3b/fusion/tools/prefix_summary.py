"""Before / after table of prefix caching from two prefix_bench + bench.sh runs (pc_tests.sh layout).

    python prefix_summary.py <runs dir> <label on> <label off> [name]
"""

import glob
import json
import os
import sys


def result(path):
    if not os.path.exists(path):
        return None
    for line in open(path):
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return None


def bench(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "bench_*.json"))):
        r = json.load(open(f))
        out[os.path.basename(f)[6:-5]] = (r.get("median_ttft_ms"), r.get("median_tpot_ms"))
    return out


def main(runs, on, off, name=None):
    name = name or on
    don, doff = os.path.join(runs, on), os.path.join(runs, off)
    print(f"### {name}")
    ron, roff = result(f"{don}/repeat.log"), result(f"{doff}/repeat.log")
    if ron and roff:
        print("\n| Prompt resent (tokens) | TTFT off | TTFT on, cold | TTFT on, warm | speed-up | greedy = off |")
        print("|---:|---:|---:|---:|---:|:---:|")
        for a, b in zip(ron["rows"], roff["rows"]):
            same = a.get("text", a["out"]) == b.get("text", b["out"])
            print(f"| {a['prompt_tokens']} | {b['ttft_warm']:.2f} s | {a['ttft_cold']:.2f} s | {a['ttft_warm']:.2f} s | "
                  f"{b['ttft_warm'] / a['ttft_warm']:.1f}x | {'yes' if same and a['identical'] else 'no'} |")
    for log, title in (("chat.log", "Chat over one document"), ("chat_think.log", "Chat, thinking on")):
        con, coff = result(f"{don}/{log}"), result(f"{doff}/{log}")
        if not (con and coff):
            continue
        print(f"\n| {title}: turn | prompt tokens | TTFT off | TTFT on | greedy = off |")
        print("|---:|---:|---:|---:|:---:|")
        for a, b in zip(con["rows"], coff["rows"]):
            same = a["reply"] == b["reply"] and a.get("reasoning") == b.get("reasoning")
            print(f"| {a['turn']} | {a['prompt_tokens']} | {b['ttft']:.2f} s | {a['ttft']:.2f} s | {'yes' if same else 'no'} |")
    bon, boff = bench(don), bench(doff)
    if bon and boff:
        print("\n| No-hit row (in/out/users) | TTFT off | TTFT on | TPOT off | TPOT on |")
        print("|---|---:|---:|---:|---:|")
        for k in bon:
            if k in boff:
                (t1, p1), (t0, p0) = bon[k], boff[k]
                print(f"| {k} | {t0:.0f} ms | {t1:.0f} ms | {p0:.2f} ms | {p1:.2f} ms |")
    print()


if __name__ == "__main__":
    main(*sys.argv[1:])
