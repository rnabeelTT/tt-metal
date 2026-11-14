# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""HYB_PLAN file from a CPU reference's routing (ref_m27.py / ref_o397.py routing.npz: ids / w [B, L, T, K]) in the
layout the overlay reads (hybrid/patch.py HybridState.plan, capture_routing.py's format): `layers` [L] and per sequence
b `p<b>` [T, L, K] expert ids (int64) and `s<b>` [T, L, K] routing weights (float32; HYB_PLAN_WEIGHT=mass ranks by them).

usage: python plan_from_routing.py <routing.npz> <plan out .npz>
"""

import sys

import numpy as np

d = np.load(sys.argv[1])
ids, w = d["ids"].astype(np.int64), d["w"].astype(np.float32)
B, L, T, K = ids.shape
out = {"layers": np.arange(L, dtype=np.int64)}
for b in range(B):
    out[f"p{b}"] = ids[b].transpose(1, 0, 2)  # [T, L, K]
    out[f"s{b}"] = w[b].transpose(1, 0, 2)
np.savez(sys.argv[2], **out)
print(f"{sys.argv[2]}: {B} sequences x {T} tokens, {L} layers, top-{K}")
