# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Per-layer slot counts C_l in {192, 224, 256} under a device-memory budget (extra slots over 192, per die: 32 slots =
32 * 1.99 MB): greedy by host experts saved per MB, on the TRAIN routing; evaluated on held-out routing in passes of
R = 4 tokens. usage: python sim_mixc_m27.py plan train_routing held_routing"""
import sys

import numpy as np

plan = np.load(sys.argv[1])
order = plan["order"]
R, MB = 4, 32 * 1.99


def host_per_layer(ids, Cs):
    B, L, T, K = ids.shape
    out = np.zeros(L)
    passes = 0
    for b in range(B):
        for t0 in range(0, T - R + 1, R):
            passes += 1
            for li in range(L):
                u = np.unique(ids[b, li, t0 : t0 + R])
                out[li] += int(
                    (np.searchsorted(np.sort(order[li, : Cs[li] - 1]), u, side="left") >= 0).sum()
                    - np.isin(u, order[li, : Cs[li] - 1]).sum()
                )
    return out / passes


tr, ho = np.load(sys.argv[2])["ids"], np.load(sys.argv[3])["ids"]
L = tr.shape[1]
h = {C: host_per_layer(tr, [C] * L) for C in (192, 224, 256)}
hh = {C: host_per_layer(ho, [C] * L) for C in (192, 224, 256)}
print(
    f"held-out host experts / pass: all 192 {hh[192].sum():.1f}, all 224 {hh[224].sum():.1f}, all 256 {hh[256].sum():.1f}"
)
for budget_gb in (1.5, 2.0, 2.5, 3.0):
    Cs = [192] * L
    left = budget_gb * 1024
    while True:
        best = None
        for li in range(L):
            if Cs[li] < 256:
                gain = h[Cs[li]][li] - h[Cs[li] + 32][li]
                if MB <= left and (best is None or gain > best[0]):
                    best = (gain, li)
        if best is None or best[0] <= 0:
            break
        Cs[best[1]] += 32
        left -= MB
    held = sum(hh[C][li] for li, C in enumerate(Cs))
    print(
        f"budget {budget_gb:.1f} GB/die: {sum(c > 192 for c in Cs)} layers raised "
        f"({sum(c == 256 for c in Cs)} to 256) -> held-out host experts / pass {held:.1f}",
        flush=True,
    )
