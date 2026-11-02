"""Per-phase MTP acceptance from a server log: the per-request summaries (MTP_STATS (phase end (request finished))),
the last 12*N of them split into N phases of 12 (realtext's 12 prompts)."""
import re, sys
lines = [l for l in open(sys.argv[1]) if "MTP_STATS (phase end (request finished))" in l]
names = sys.argv[2:]
n = 12
rows = lines[-n * len(names):]
for i, name in enumerate(names):
    P = T = W = 0
    H = None
    for l in rows[i * n:(i + 1) * n]:
        m = re.search(r"(\d+) passes, (\d+) tokens.*hist \[([^]]*)\], ([\d.]+) ms per pass", l)
        p, t, h, ms = int(m[1]), int(m[2]), [int(x) for x in m[3].split(",")], float(m[4])
        P += p; T += t; W += p * ms
        H = h if H is None else [a + b for a, b in zip(H, h)]
    print(f"MTPACC {name}: {len(rows[i*n:(i+1)*n])} requests, {P} passes, {T} tokens = {T / P:.2f} per pass, hist {H}, "
          f"{W / P:.1f} ms per pass (device steps)")
