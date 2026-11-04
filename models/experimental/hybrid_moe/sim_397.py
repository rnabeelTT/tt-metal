# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Expert-cache policies on recorded 397B decode routing (run_generate HYB_RECORD routing.npz files), leave-one-prompt-out:
the static set of a prompt comes from the other prompts' routing frequency. usage: python sim_397.py routing.npz [...]"""
import json
import sys
from collections import OrderedDict

import numpy as np

E, K = 512, 10


def load(paths):
    prompts = {}
    for p in paths:
        z = np.load(p)
        for k in z.files:
            if k.startswith("p"):
                prompts[f"{p}:{k}"] = z[k]  # [T, L, K]
    return prompts


def freq(arrs, L):
    f = np.zeros((L, E))
    for a in arrs:
        for li in range(L):
            np.add.at(f[li], a[:, li, :].ravel(), 1)
    return f


def run(a, f, C, D, promote):
    """C static slots (top frequency) + D dynamic LRU slots, at most `promote` promotions per layer per token
    (promotions land after the token, i.e. help later tokens). Returns hit rate."""
    T, L, _ = a.shape
    hits = tot = 0
    for li in range(L):
        static = set(np.argsort(-f[li], kind="stable")[:C].tolist())
        lru = OrderedDict()
        for t in range(T):
            ids = a[t, li].tolist()
            miss = []
            for e in ids:
                tot += 1
                if e in static:
                    hits += 1
                elif e in lru:
                    hits += 1
                    lru.move_to_end(e)
                else:
                    miss.append(e)
            for e in miss[:promote] if D else []:
                if len(lru) >= D:
                    lru.popitem(last=False)
                lru[e] = True
    return hits / tot


def main():
    prompts = load(sys.argv[1:])
    names = sorted(prompts)
    L = next(iter(prompts.values())).shape[1]
    # memory: one bfp8 dynamic slot = two bfp4 static slots; budget = 272 static-slot equivalents
    configs = [("static", 272, 0, 0), ("random-equiv", None, 0, 0)]
    for D in (16, 32, 64, 96):
        for pr in (1, 2, 10):
            configs.append((f"C{272 - 2 * D}+D{D} promote{pr}", 272 - 2 * D, D, pr))
    configs.append(("bfp4-dyn C208+D64 promote2 (if bfp4 slots were writable)", 208, 64, 2))
    configs.append(("bfp4-dyn C144+D128 promote2", 144, 128, 2))
    out = {}
    for label, C, D, pr in configs:
        if C is None:
            out[label] = 272 / E
            continue
        rates = []
        for n in names:
            f = freq([prompts[m] for m in names if m != n], L)
            rates.append(run(prompts[n], f, C, D, pr))
        out[label] = round(float(np.mean(rates)), 4)
        print(f"{label:60s} {out[label]:.3f}  (per prompt {min(rates):.3f}-{max(rates):.3f})", flush=True)
    json.dump(out, open("runs/sim_397.json", "w"), indent=1)


if __name__ == "__main__":
    main()
