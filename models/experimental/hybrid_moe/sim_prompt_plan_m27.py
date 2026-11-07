# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Per-request plan: the static set (top C-1 by train mass) with S experts per layer swapped at prefill for the cold
experts with the most routing mass over the request's first P tokens (stand-in for the prompt); victims = the static
experts with the least train mass among those the prompt did not use. Replays the rest of each held-out sequence in
DFlash-like passes of R tokens: host experts per pass = the union of the pass's picks not on the device.
usage: python sim_prompt_plan_m27.py <plan.npz> <held routing.npz>"""
import sys

import numpy as np

plan, held = np.load(sys.argv[1]), np.load(sys.argv[2])
order, mass, ids, w = plan["order"], plan["mass"], held["ids"], held["w"]
B, L, T, K = ids.shape
C1, R = 191, 4
for P in (64, 256):
    for S in (0, 8, 16, 32):
        host = passes = 0
        for b in range(B):
            dev = np.zeros((L, 256), bool)
            dev[np.arange(L)[:, None], order[:, :C1]] = True
            if S:
                for li in range(L):
                    pm = np.zeros(256)
                    np.add.at(pm, ids[b, li, :P].ravel(), w[b, li, :P].ravel())
                    cand = [e for e in np.argsort(-pm) if not dev[li, e] and pm[e] > 0][:S]
                    vict = [e for e in order[li, :C1][::-1] if pm[e] == 0][: len(cand)]
                    dev[li, vict] = False
                    dev[li, cand[: len(vict)]] = True
            for t0 in range(P, T - R + 1, R):
                passes += 1
                for li in range(L):
                    u = np.unique(ids[b, li, t0 : t0 + R])
                    host += int((~dev[li, u]).sum())
        print(f"prompt {P:3d} tokens, swap {S:2d}/layer: host experts / pass {host / passes:6.1f}", flush=True)
