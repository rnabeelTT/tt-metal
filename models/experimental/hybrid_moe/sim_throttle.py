# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Dynamic (global LRU) cache when PCIe can promote only a fraction p of misses: each miss is promoted with
probability p (deterministic seed). Hit rate vs capacity and p, on the 122B routing (leave-one-out warm start)."""
import collections, json, sys
import numpy as np
from sim_cache import load, freq, ranked_hot

arrs = load("routing/routing.npz"); L = arrs[0].shape[1]; E = 256
res = {}
for cap in (0.4, 0.5, 0.6, 0.7):
    budget = int(cap * L * E)
    for p in (1.0, 0.9, 0.75, 0.5, 0.25):
        rng = np.random.default_rng(0); hits = tot = 0
        for i, a in enumerate(arrs):
            rk = ranked_hot(freq([b for j, b in enumerate(arrs) if j != i], L), budget)
            lru = collections.OrderedDict((k, None) for k in rk)
            for s in range(a.shape[0]):
                for li in range(L):
                    for e in a[s, li]:
                        key = (li, int(e)); tot += 1
                        if key in lru:
                            hits += 1; lru.move_to_end(key)
                        elif rng.random() < p:
                            lru[key] = None
                            if len(lru) > budget: lru.popitem(last=False)
        res[f"{cap}_{p}"] = round(hits / tot, 4)
    print(cap, {p: res[f"{cap}_{p}"] for p in (1.0, 0.9, 0.75, 0.5, 0.25)}, flush=True)
json.dump(res, open("routing/sim_throttle.json", "w"), indent=1)
