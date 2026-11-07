# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""CPU / PCIe split of the host experts with D device slots per layer taken from the 192 (static = top 191 - D by
train mass). Per DFlash-like pass of R tokens and per layer: the union of picks; dyn hits are free; of the remaining
misses, up to P go over PCIe into dyn slots (LRU; they are computed on the device and stay for later passes), the rest
run on the CPU. Layer time = max(cpu * c_ms, pcie * p_ms) (concurrent rates). usage: python sim_split_m27.py plan held"""
import sys
from collections import OrderedDict

import numpy as np

plan, held = np.load(sys.argv[1]), np.load(sys.argv[2])
order, ids = plan["order"], held["ids"]
B, L, T, K = ids.shape
R, c_ms, p_ms = 4, 0.30, 0.60
print("baseline (no split): CPU 0.25 ms/expert alone")
for D, P in ((0, 0), (2, 1), (2, 2), (4, 2), (8, 2), (16, 2), (16, 3), (32, 3), (32, 4)):
    static = np.zeros((L, 256), bool)
    static[np.arange(L)[:, None], order[:, : 191 - D]] = True
    cpu = pcie = passes = 0
    t_host = 0.0
    for b in range(B):
        lru = [OrderedDict() for _ in range(L)]
        for t0 in range(0, T - R + 1, R):
            passes += 1
            for li in range(L):
                u = np.unique(ids[b, li, t0 : t0 + R])
                miss = []
                for e in u[~static[li, u]]:
                    if e in lru[li]:
                        lru[li].move_to_end(e)
                    else:
                        miss.append(e)
                up = miss[: min(P, D)]
                for e in up:
                    lru[li][e] = 1
                    if len(lru[li]) > D:
                        lru[li].popitem(last=False)
                nc = len(miss) - len(up)
                cpu += nc
                pcie += len(up)
                t_host += nc * 0.25 if not up else max(nc * c_ms, len(up) * p_ms)
    print(
        f"dyn slots {D:2d}, PCIe <= {P}/layer: CPU experts/pass {cpu / passes:6.1f}  PCIe/pass {pcie / passes:5.1f}  "
        f"host critical ms/pass {t_host / passes:5.1f}",
        flush=True,
    )
