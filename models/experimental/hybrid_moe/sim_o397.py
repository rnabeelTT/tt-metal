# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Ornith-1.5-397B (512 experts, top-10, H 4096, I 1024) copy of sim_m27.py. Hit-rate simulation for a static per-layer
expert plan from ref_o397.py routing: rank each layer's experts by routed
weight on the train set, keep the top C on the device, and replay held-out routing (one token per step, as in decode).

usage: python sim_o397.py <train routing.npz> <held routing.npz> [plan out .npz]
Reports, per C: the share of picks that hit, the routing mass of the misses, misses per token, and the host bytes
per token at skip 0 (int4 group-32 experts: 3 * H * I / 2 weight bytes + 4 bytes per 32 weights of scale/zero).
"""

import sys

import numpy as np

H, I = 4096, 1024
EXPERT_BYTES = 3 * H * I // 2 + 3 * H * I // 32 * 4


def load(p):
    d = np.load(p)
    return d["names"], d["ids"].astype(np.int64), d["w"].astype(np.float32)


def main():
    _, tr_ids, tr_w = load(sys.argv[1])
    names, ho_ids, ho_w = load(sys.argv[2])
    B, L, T, K = tr_ids.shape
    E = 512
    mass = np.zeros((L, E))
    for li in range(L):
        np.add.at(mass[li], tr_ids[:, li].ravel(), tr_w[:, li].ravel())
    order = np.argsort(-mass, axis=1)
    if len(sys.argv) > 3:  # per-layer experts ranked by train routing mass (order[:, :C] = the device set)
        np.savez(sys.argv[3], order=order, mass=mass)
    print(f"train {B} x {T} tokens, held-out {ho_ids.shape[0]} x {ho_ids.shape[2]}; {L} layers")
    for C in (128, 192, 224, 256, 288, 320, 352, 384, 448):
        res = np.zeros((L, E), dtype=bool)
        res[np.arange(L)[:, None], order[:, :C]] = True
        hit = res[np.arange(L)[None, :, None, None], ho_ids]  # [B, L, T, K]
        miss_w = (ho_w * ~hit).sum(-1)  # [B, L, T]
        per_tok = (~hit).sum((1, 3)).astype(np.float64)  # misses per token over all layers [B, T]
        mb = per_tok.mean() * EXPERT_BYTES / 1e6
        per_seq = ", ".join(f"{n[:10]} {hit[b].mean():.3f}" for b, n in enumerate(names))
        print(f"C={C:3d} ({C / E:.0%}): hit {hit.mean():.4f}  miss mass/layer {miss_w.mean():.4f}  "
              f"misses/token {per_tok.mean():.1f}  host MB/token {mb:.0f}  worst layer hit {hit.mean((0, 2, 3)).min():.3f}")
        print(f"        {per_seq}")
    in_hit = np.take_along_axis(np.argsort(order, 1), tr_ids.transpose(1, 0, 2, 3).reshape(L, -1), 1) < 256
    print(f"in-sample hit at C=256: {in_hit.mean():.4f}")


if __name__ == "__main__":
    main()
