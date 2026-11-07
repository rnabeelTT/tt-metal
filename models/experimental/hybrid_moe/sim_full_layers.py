# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""HYB_FULL_LAYERS: which layers to hold entirely on the device (all E experts) under a slot budget. Ranks layers by
their host experts per pass at C slots on the TRAIN routing (passes of R tokens: the union of picks), reports the
held-out host experts per pass for the top n, and prints the list for n.
usage: python sim_full_layers.py <plan.npz> <train routing.npz> <held routing.npz> <n> [C=192] [R=4]"""
import sys

import numpy as np

plan = np.load(sys.argv[1])
order = plan["order"]
n = int(sys.argv[4])
C = int(sys.argv[5]) if len(sys.argv) > 5 else 192
R = int(sys.argv[6]) if len(sys.argv) > 6 else 4


def per_layer(ids):
    B, L, T, K = ids.shape
    out, passes = np.zeros(L), 0
    for b in range(B):
        for t0 in range(0, T - R + 1, R):
            passes += 1
            for li in range(L):
                u = np.unique(ids[b, li, t0 : t0 + R])
                out[li] += (~np.isin(u, order[li, : C - 1])).sum()
    return out / passes


tr = per_layer(np.load(sys.argv[2])["ids"])
ho = per_layer(np.load(sys.argv[3])["ids"])
rank = np.argsort(-tr)
for k in sorted({0, n // 2, n}):
    print(f"{k:2d} full layers: held-out host experts / pass {ho.sum() - ho[rank[:k]].sum():.1f}")
print(",".join(str(int(v)) for v in sorted(rank[:n])))
