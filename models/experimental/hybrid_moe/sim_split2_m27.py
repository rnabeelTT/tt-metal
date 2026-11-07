# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""FreeToken-style hybrid on M2.7: S static + D LRU slots per layer (S + D = 191), passes of R = 4 tokens; per layer
the misses (picks neither static nor in the LRU) are split: a share f is fetched over PCIe into the LRU (computed on the
device, kept for later passes), the rest computed on the CPU. Fetch count = f * misses rounded stochastically. Layer
host time = max(cpu * c_ms, fetched * p_ms) with concurrent per-expert costs (both read host DRAM).
usage: python sim_split2_m27.py plan.npz held_routing.npz"""
import sys
from collections import OrderedDict

import numpy as np

order = np.load(sys.argv[1])["order"]
ids = np.load(sys.argv[2])["ids"]
B, L, T, K = ids.shape
R = 4
rng = np.random.default_rng(0)
print("now: S=191 D=0 CPU only, 0.25 ms/expert -> ~53.5 ms/pass")
for c_ms, p_ms in ((0.30, 0.45), (0.30, 0.60)):
    for D in (16, 32, 64):
        for f in (0.3, 0.4, 0.5):
            S = 191 - D
            static = np.zeros((L, 256), bool)
            static[np.arange(L)[:, None], order[:, :S]] = True
            t = cpu = fet = 0.0
            passes = 0
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
                        x = f * len(miss)
                        nf = int(x) + (rng.random() < x - int(x))
                        for e in miss[:nf]:
                            lru[li][e] = 1
                            if len(lru[li]) > D:
                                lru[li].popitem(last=False)
                        nc = len(miss) - nf
                        cpu += nc
                        fet += nf
                        t += max(nc * c_ms, nf * p_ms)
            print(
                f"c {c_ms} p {p_ms} | static {S} + LRU {D:2d}, f {f}: CPU {cpu / passes:5.1f} fetch {fet / passes:5.1f}"
                f" /pass -> host {t / passes:5.1f} ms/pass",
                flush=True,
            )
