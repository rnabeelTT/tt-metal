# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""hostk_moe_csr (bf16 prefill kernel) vs hostk_moe (fp32, MAXR-row tiles, the packed prefill path) on a real int4
library layer: max / relative error and time for one prefill chunk's missed pairs.
usage: python tests/bench_hostk_csr.py [library dir] [layer] [rows] [pairs] [threads]"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hybrid import hostk  # noqa: E402

lib_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/.cache/hybrid-moe/inkling_int4_fb191")
layer = int(sys.argv[2]) if len(sys.argv) > 2 else 10
R = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
P = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
nt = int(sys.argv[5]) if len(sys.argv) > 5 else 8
lib = torch.load(os.path.join(lib_dir, f"layer_{layer}.pt"), weights_only=False)
lib = {k: (v.contiguous() if hasattr(v, "contiguous") else v) for k, v in lib.items()}
E = lib["gu_p"].shape[0]
H = lib["gu_p"].shape[2] * 2
hostk.load(nt)
torch.manual_seed(0)
X = (torch.randn(R, H) * 0.5).to(torch.bfloat16)
# skewed expert popularity (a few experts take many tokens), each pair a distinct (row, expert)
pop = torch.distributions.Dirichlet(torch.full((E,), 0.3)).sample()
e_all = torch.multinomial(pop, P, replacement=True)
r_all = torch.randint(0, R, (P,))
key = torch.unique(e_all * R + r_all)
e_all, r_all = key // R, key % R
w_all = torch.rand(len(e_all)) * 0.3
order = torch.argsort(e_all, stable=True)
e_all, r_all, w_all = e_all[order], r_all[order], w_all[order]
ex, cnt = torch.unique(e_all, return_counts=True)
print(f"layer {layer}: {len(e_all)} pairs over {len(ex)} experts (max {int(cnt.max())} rows), R {R}, threads {nt}")


def ref_packed():
    out = torch.zeros(R, H)
    Xf = X.float()
    M = hostk.MAXR
    for c0 in range(0, len(e_all), M):
        r, e, w = r_all[c0 : c0 + M], e_all[c0 : c0 + M], w_all[c0 : c0 + M]
        exs, inv = torch.unique(e, return_inverse=True)
        W = torch.zeros(len(exs), len(r))
        W[inv, torch.arange(len(r))] = w
        oc = torch.zeros(len(r), H)
        hostk.moe(lib, Xf[r].contiguous(), exs.contiguous(), W, oc)
        out.index_add_(0, r, oc)
    return out


def csr():
    out = torch.zeros(R, H)
    off = torch.zeros(len(ex) + 1, dtype=torch.int32)
    off[1:] = torch.cumsum(cnt, 0).to(torch.int32)
    hostk.moe_csr(
        lib, X.contiguous(), ex.contiguous(), off, r_all.to(torch.int32).contiguous(), w_all.float().contiguous(), out
    )
    return out


for name, f in (("packed fp32", ref_packed), ("csr bf16", csr)):
    f()
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        o = f()
        ts.append(time.perf_counter() - t0)
    print(f"{name:12s} {1e3 * min(ts):7.1f} ms")
    if name == "packed fp32":
        ref = o
a = csr()
d = (a - ref).abs()
print(
    f"max abs err {float(d.max()):.3e}, rel (fro) {float(d.norm() / ref.norm()):.3e}, "
    f"cos {float(torch.nn.functional.cosine_similarity(a.flatten(), ref.flatten(), dim=0)):.7f}"
)
