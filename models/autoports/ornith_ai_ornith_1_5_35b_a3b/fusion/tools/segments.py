"""Per-layer device time of the signposted traced decode windows (device 0), split into boundary / mixer / MoE.

    python segments.py <ops_perf csv> [iters] [--ops]

Segments are found from the collectives, so they survive op-count changes: the boundary is the attention-norm
all-gather + norm + its sharded->interleaved copy, the mixer reduce-scatter + residual add, the ffn-norm all-gather +
norm + copy, and the MoE reduce-scatter + residual add; the mixer is everything between the first boundary and the
first reduce-scatter; the MoE everything between the ffn-norm block and the last reduce-scatter. The first op's gap
(the idle time between trace replays) is capped at 20 us. --ops prints the MoE ops of one step.
"""

import csv
import re
import sys


def windows(path):
    rows = list(csv.DictReader(open(path)))
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
    return wins


def name(x):
    return re.sub("DeviceOperation", "", x["OP CODE"])


def segments(step):
    names = [name(x) for x in step]
    ag = [i for i, n in enumerate(names) if n.startswith("AllGather")]
    rs = [i for i, n in enumerate(names) if n.startswith("ReduceScatter")]
    assert len(ag) == 2 and len(rs) == 2, (ag, rs)
    first = ag[1] + 3  # ffn all-gather, norm, sharded->interleaved
    boundary = [0, 1, 2, rs[0], rs[0] + 1, ag[1], ag[1] + 1, ag[1] + 2] + list(range(rs[1], len(step)))
    return {
        "boundary": boundary,
        "mixer": list(range(3, rs[0])),
        "moe": list(range(first, rs[1])),
    }


def us(x, k):
    try:
        return float(x[k] or 0) / 1e3
    except ValueError:
        return 0.0


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    iters = int(args[1]) if len(args) > 1 else 5
    for label, w in zip(["gdn", "attn"], windows(args[0])):
        n = len(w) // iters
        print(f"== {label}: {n} ops/step")
        tot = 0.0
        for seg, idx in segments(w[:n]).items():
            tk = to = 0.0
            for it in range(1, iters):
                step = w[it * n : (it + 1) * n]
                for i in idx:
                    tk += us(step[i], "DEVICE KERNEL DURATION [ns]")
                    g = us(step[i], "OP TO OP LATENCY [ns]")
                    to += min(g, 20) if i == 0 else g
            k, g = tk / (iters - 1), to / (iters - 1)
            tot += k + g
            print(f"  {seg:9s} ops {len(idx):3d} kernel {k:6.1f} us gaps {g:6.1f} us total {k + g:6.1f} us")
        print(f"  {'layer':9s} {'':7s} {'':6s}{'':10s}{'':14s} total {tot:6.1f} us")
        if "--ops" in sys.argv:
            step = w[2 * n : 3 * n]
            for i in segments(step)["moe"]:
                x = step[i]
                print(
                    f"    {i:3d} {name(x)[:30]:30s} {us(x, 'DEVICE KERNEL DURATION [ns]'):6.1f} us "
                    f"gap {us(x, 'OP TO OP LATENCY [ns]'):5.1f} cores {x.get('CORE COUNT', '')}"
                )


if __name__ == "__main__":
    main()
