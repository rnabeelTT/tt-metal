# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Hybrid-MoE POC: replay recorded decode routing (routing.npz from capture_routing.py) against expert caches.

Capacity is a fraction of all (layer, expert) pairs held on device, shared across layers (one global budget).
Policies:
  static      hot set = most frequently routed (layer, expert) pairs on the OTHER prompts (leave-one-out)
  lru         per-layer LRU, per-layer capacity = cap * E, warm-started from the static hot set
  static+lru  (1-d) of the budget static, d dynamic (global LRU over the remaining pairs), warm-started
Hit rate = fraction of routed (token, layer, expert) picks resident when routed. A miss is inserted after the step
(it was fetched / promoted); in static it is not.
"""

import collections
import json
import sys

import numpy as np

E = 256


def load(path):
    z = np.load(path)
    prompts = sorted([k for k in z.files if k.startswith("p")], key=lambda s: int(s[1:]))
    return [z[k] for k in prompts]  # each [steps, layers, k]


def freq(arrs, L):
    f = np.zeros((L, E))
    for a in arrs:
        for li in range(L):
            np.add.at(f[li], a[:, li, :].ravel(), 1)
    return f


def hot_set(f, budget):
    flat = np.argsort(-f.ravel(), kind="stable")[:budget]
    return {(int(i) // E, int(i) % E) for i in flat}


def sim_static(a, hot):
    hits = tot = 0
    for li in range(a.shape[1]):
        col = a[:, li, :]
        tot += col.size
        hits += sum(1 for e in col.ravel() if (li, int(e)) in hot)
    return hits, tot


def sim_dynamic(a, hot, budget, dyn_frac):
    """static part = top (1-dyn_frac)*budget of hot (hot is ordered by insertion = rank); dynamic part = global LRU."""
    static_n = int(budget * (1 - dyn_frac))
    ranked = list(hot)
    static = set(ranked[:static_n])
    lru = collections.OrderedDict((k, None) for k in ranked[static_n:budget])
    cap = budget - static_n
    hits = tot = 0
    per_step_miss = []
    for s in range(a.shape[0]):
        miss_this = 0
        for li in range(a.shape[1]):
            for e in a[s, li]:
                key = (li, int(e))
                tot += 1
                if key in static:
                    hits += 1
                elif key in lru:
                    hits += 1
                    lru.move_to_end(key)
                else:
                    miss_this += 1
                    if cap:
                        lru[key] = None
                        if len(lru) > cap:
                            lru.popitem(last=False)
        per_step_miss.append(miss_this)
    return hits, tot, per_step_miss


def ranked_hot(f, budget):
    flat = np.argsort(-f.ravel(), kind="stable")[:budget]
    return [(int(i) // E, int(i) % E) for i in flat]


def main():
    arrs = load(sys.argv[1])
    L, K = arrs[0].shape[1], arrs[0].shape[2]
    caps = [0.3, 0.4, 0.5, 0.6, 0.7]
    res = {"prompts": len(arrs), "tokens": int(sum(a.shape[0] for a in arrs)), "layers": L, "k": K, "results": {}}
    for cap in caps:
        budget = int(cap * L * E)
        row = collections.defaultdict(lambda: [0, 0])
        for i, a in enumerate(arrs):
            others = [b for j, b in enumerate(arrs) if j != i]
            f = freq(others, L)
            rk = ranked_hot(f, budget)
            h, t = sim_static(a, set(rk))
            row["static"][0] += h
            row["static"][1] += t
            for d in (0.1, 0.25, 0.5, 1.0):
                h, t, _ = sim_dynamic(a, rk, budget, d)
                row[f"static+lru_d{d}"][0] += h
                row[f"static+lru_d{d}"][1] += t
        res["results"][cap] = {k: round(v[0] / v[1], 4) for k, v in row.items()}
        print(cap, res["results"][cap], flush=True)
    # self-similarity: overlap of consecutive steps' expert sets per layer
    ov = []
    for a in arrs:
        for s in range(1, a.shape[0]):
            for li in range(L):
                ov.append(len(set(a[s, li]) & set(a[s - 1, li])) / K)
    res["consecutive_step_overlap"] = round(float(np.mean(ov)), 4)
    f_all = freq(arrs, L)
    p = np.sort(f_all, axis=1)[:, ::-1]
    cum = np.cumsum(p, axis=1) / p.sum(axis=1, keepdims=True)
    res["per_layer_mass_in_top_frac"] = {
        str(fr): round(float(cum[:, int(fr * E) - 1].mean()), 4) for fr in (0.1, 0.25, 0.5, 0.75)
    }
    print(json.dumps({k: v for k, v in res.items() if k != "results"}, indent=1))
    json.dump(res, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
