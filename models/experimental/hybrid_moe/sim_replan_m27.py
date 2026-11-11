# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Per-request re-plan on top of the served policy: hybrid layers only (HYB_FULL_LAYERS excluded), static top S0 of
the train plan, D LRU slots with the fetch rule (f x misses, at most F per layer step, most recent first), passes of R
rows. At prefill end, `swap` static experts per layer are replaced by the cold experts with the most routing mass over
the request's first P tokens (victims: the static experts the prompt did not use, least train mass first). Prints CPU
experts per pass (summed over the hybrid layers) and the static swaps (copies) the re-plan costs.
usage: python sim_replan_m27.py <plan.npz> <held routing.npz> <full layers csv>"""
import sys
from collections import OrderedDict

import numpy as np

plan, held = np.load(sys.argv[1]), np.load(sys.argv[2])
full = {int(v) for v in sys.argv[3].split(",")}
order, ids, w = plan["order"], held["ids"], held["w"]
B, L, T, K = ids.shape
S0, D, F, f, R = 159, 32, 2, 0.3, 4
hyb = [li for li in range(L) if li not in full]
rng = np.random.default_rng(0)
VMIN = int(sys.argv[4]) if len(sys.argv) > 4 else 0
RATIO = (
    float(sys.argv[5]) if len(sys.argv) > 5 else 2.0
)  # swap only when the candidate has RATIO x the victim's mass  # victims only from static ranks >= VMIN (host copies held)
MODES = [("base", 0, 0), ("swap32", 32, 0)]
for P in (64, 512):
    for name, swap, seed in MODES:
        cpu = passes = copies = vfetch = 0
        for b in range(B):
            for li in hyb:
                st = np.zeros(256, bool)
                st[order[li, :S0]] = True
                if swap:
                    pm = np.zeros(256)
                    np.add.at(pm, ids[b, li, :P].ravel(), w[b, li, :P].ravel())
                    cand = [e for e in np.argsort(-pm) if not st[e] and pm[e] > 0][:swap]
                    vs = order[li, VMIN:S0]
                    vict = [int(e) for e in vs[np.lexsort((np.arange(len(vs))[::-1], pm[vs]))]]  # least mass
                    pairs = [(c, v) for c, v in zip(cand, vict) if pm[c] > RATIO * pm[v]]
                    for c, v in pairs:
                        st[v] = False
                        st[c] = True
                    copies += len(pairs)
                    vmiss_set = {v for _, v in pairs}
                lru = OrderedDict()
                if seed:  # the prompt's heaviest experts outside the static set start in the LRU slots
                    pm = np.zeros(256)
                    np.add.at(pm, ids[b, li, :P].ravel(), w[b, li, :P].ravel())
                    for e in [int(e) for e in np.argsort(pm) if not st[e] and pm[e] > 0][-D:]:
                        lru[e] = True
                        copies += 1
                for t0 in range(P, T - R + 1, R):
                    if li == hyb[0]:
                        passes += 1
                    u = [int(e) for e in np.unique(ids[b, li, t0 : t0 + R]) if not st[e]]
                    miss = [e for e in u if e not in lru]
                    for e in u:
                        if e in lru:
                            lru.move_to_end(e)
                    n = f * len(miss)
                    n = min(F, int(n) + (rng.random() < n - int(n)))
                    for e in miss[:n]:
                        lru[e] = True
                        if len(lru) > D:
                            lru.popitem(last=False)
                    cpu += len(miss) - n
                    if swap:
                        vfetch += sum(1 for e in miss if e in vmiss_set)
        print(
            f"prompt {P:3d}, {name:12s}: CPU experts / pass {cpu / passes:6.1f}, "
            f"copies / request {copies / B:6.0f}, victim misses / pass {vfetch / max(1, passes):5.1f}",
            flush=True,
        )
