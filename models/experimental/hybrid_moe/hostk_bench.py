# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host int4 expert kernel (hybrid/hostk) cost per expert vs active rows, on one layer of the served library: the
input to the prefill CPU / PCIe split. No device. usage: python hostk_bench.py [lib dir] [layer]
env: HB_THREADS (12), HB_EXPERTS (48)"""

import os
import sys
import time

import torch

from hybrid import hostk

lib_dir = sys.argv[1] if len(sys.argv) > 1 else "/hyblib/minimax_m27_int4_s152"
layer = int(sys.argv[2]) if len(sys.argv) > 2 else 30
nt = int(os.environ.get("HB_THREADS", "12"))
NE = int(os.environ.get("HB_EXPERTS", "48"))
hostk.load(nt)
d = torch.load(os.path.join(lib_dir, f"layer_{layer}.pt"), mmap=True, weights_only=True)
n_lib = int(d["gu_p"].shape[0])
idx = torch.arange(min(NE, n_lib))
lib = {k: d[k].index_select(0, idx).contiguous() for k in ("gu_p", "gu_sz", "dn_p", "dn_sz")}
lib["gs"] = d["gs"]
H = int(d["dn_p"].shape[1]) if d["dn_p"].dim() == 3 else 3072
print(
    f"layer {layer}: {n_lib} library experts, gu_p {tuple(d['gu_p'].shape)} dn_p {tuple(d['dn_p'].shape)}, threads {nt}"
)
bytes_per_expert = sum(lib[k][0].numel() * lib[k].element_size() for k in ("gu_p", "gu_sz", "dn_p", "dn_sz"))
H = 3072
E = len(idx)


def run(rows_per_expert, R=hostk.MAXR):
    """E experts in calls of R rows; each expert active on rows_per_expert rows of its call (block layout)."""
    per_call = max(1, R // rows_per_expert)
    t = 0.0
    for c0 in range(0, E, per_call):
        ex = list(range(c0, min(E, c0 + per_call)))
        X = torch.randn(R, H)
        W = torch.zeros(len(ex), R)
        for i in range(len(ex)):
            W[i, i * rows_per_expert : (i + 1) * rows_per_expert] = 0.125
        out = torch.zeros(R, H)
        t0 = time.perf_counter()
        hostk.moe(lib, X, torch.tensor(ex, dtype=torch.int64), W, out)
        t += time.perf_counter() - t0
    return t


run(1)
for rpe in (1, 2, 4, 8, 16):
    ts = [run(rpe) for _ in range(3)]
    t = min(ts)
    print(
        f"HK rows/expert {rpe:2d}: {t / E * 1e3:.3f} ms per expert ({E} experts, "
        f"{bytes_per_expert * E / t / 1e9:.1f} GB/s of weights)"
    )
