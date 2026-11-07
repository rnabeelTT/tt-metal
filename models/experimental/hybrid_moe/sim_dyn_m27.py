# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Static plan (top C by train mass) + D dynamic LRU slots per layer, replayed on held-out routing in DFlash-like
passes of R consecutive tokens: per pass, the union of the R tokens' picks per layer; a pick that is neither static nor
dynamic is a host expert (computed on the CPU this pass) and is then promoted into the LRU (one PCIe write).
usage: python sim_dyn_m27.py <plan.npz> <held routing.npz>"""
import sys
from collections import OrderedDict

import numpy as np

plan, held = np.load(sys.argv[1]), np.load(sys.argv[2])
order, ids = plan["order"], held["ids"]  # [L, E], [B, L, T, K]
B, L, T, K = ids.shape
for C in (192,):
    static = np.zeros((L, 256), bool)
    static[np.arange(L)[:, None], order[:, : C - 1]] = True  # C - 1 experts + the sink slot
    for R in (1, 4):
        for D in (0, 8, 16, 32, 64):
            host = prom = passes = 0
            for b in range(B):
                lru = [OrderedDict() for _ in range(L)]
                for t0 in range(0, T - R + 1, R):
                    passes += 1
                    for li in range(L):
                        u = np.unique(ids[b, li, t0 : t0 + R])
                        for e in u[~static[li, u]]:
                            if e in lru[li]:
                                lru[li].move_to_end(e)
                                continue
                            host += 1
                            if D:
                                lru[li][e] = 1
                                prom += 1
                                if len(lru[li]) > D:
                                    lru[li].popitem(last=False)
            print(
                f"C={C} R={R} D={D:2d}: host experts / pass {host / passes:6.1f}   promotions / pass {prom / passes:6.1f}",
                flush=True,
            )
