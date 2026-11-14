# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Inkling-Small routing -> the overlay's formats + a static-plan hit-rate simulation (the sim_m27.py of Inkling).

ref_inkling.py's routing.npz holds the 40 MoE layers only (`layers` = 2..41), the 6 routed picks (ids, w) and the 2
shared gammas (shared), all x route_scale x global_scale. The port runs every MoE layer as 8 weighted experts of 258
(the shared ones are experts 256, 257; weights normalised to sum 1, the scale folded into the experts). This script:

  * writes <out>_routing.npz in the overlay's layout, indexed by MODEL layer (42 rows; the dense layers 0-1 get the
    two shared ids so no tool counts misses there): ids [B, 42, T, 8] (picks then 256, 257), w [B, 42, T, 8]
    normalised -- sim_full_layers.py and plan_from_routing.py read it as they read M2.7's;
  * writes <out>.npz: order [42, 258] per layer by train routing mass with the 2 shared experts FIRST (always on the
    device: every token uses them), mass [42, 258];
  * prints, per slot count C, the held-out hit rate, miss mass and host bytes per token.

usage: python sim_inkling.py <train routing.npz> <held routing.npz> <plan out .npz>   (writes <plan>_routing.npz too,
       for the train set, and <plan>_held_routing.npz)
"""

import sys

import numpy as np

H, I = 4096, 2048
N_LAYERS, R, S = 42, 256, 2
E = R + S
EXPERT_BYTES = 3 * H * I // 2 + 3 * H * I // 32 * 4  # int4 group-32 weights + bf16 scale/zero per group


def to_model_layers(path):
    d = np.load(path)
    layers = d["layers"].astype(np.int64)
    ids6, w6, sh = d["ids"].astype(np.int64), d["w"].astype(np.float32), d["shared"].astype(np.float32)
    B, Lm, T, K = ids6.shape
    ids = np.empty((B, N_LAYERS, T, K + S), dtype=np.int64)
    w = np.zeros((B, N_LAYERS, T, K + S), dtype=np.float32)
    ids[:] = R  # dense layers: point every pick at a shared (always resident) expert
    ids[..., -1] = R + 1
    tot = w6.sum(-1, keepdims=True) + sh.sum(-1, keepdims=True)
    ids[:, layers, :, :K] = ids6
    ids[:, layers, :, K:] = np.arange(R, E)
    w[:, layers, :, :K] = w6 / tot
    w[:, layers, :, K:] = sh / tot
    return d["names"], layers, ids, w


def main():
    _, layers, tr_ids, tr_w = to_model_layers(sys.argv[1])
    names, _, ho_ids, ho_w = to_model_layers(sys.argv[2])
    out = sys.argv[3]
    base = out[: -len(".npz")] if out.endswith(".npz") else out
    np.savez(base + "_routing.npz", names=np.array([]), ids=tr_ids.astype(np.uint16), w=tr_w.astype(np.float16))
    np.savez(base + "_held_routing.npz", names=names, ids=ho_ids.astype(np.uint16), w=ho_w.astype(np.float16))
    B, L, T, K = tr_ids.shape
    mass = np.zeros((L, E))
    for li in layers:
        np.add.at(mass[li], tr_ids[:, li].ravel(), tr_w[:, li].ravel())
    mass[:, R:] = np.inf  # the shared experts rank first everywhere
    order = np.argsort(-mass, axis=1, kind="stable")
    np.savez(out, order=order, mass=np.where(np.isinf(mass), -1.0, mass), layers=layers)
    print(f"train {B} x {T} tokens, held-out {ho_ids.shape[0]} x {ho_ids.shape[2]}; {len(layers)} MoE layers of {L}")
    lm = np.asarray(layers)
    for C in (130, 162, 194, 202, 210, 226, 242, 258):
        res = np.zeros((L, E), dtype=bool)
        res[np.arange(L)[:, None], order[:, :C]] = True
        hit = res[np.arange(L)[None, :, None, None], ho_ids][:, lm]  # [B, Lm, T, 8]
        routed_hit = hit[..., : K - S]
        miss_w = (ho_w[:, lm] * ~hit).sum(-1)
        per_tok = (~hit).sum((1, 3)).astype(np.float64)
        mb = per_tok.mean() * EXPERT_BYTES / 1e6
        print(f"C={C:3d} ({(C - S) / R:.0%} of routed): routed-pick hit {routed_hit.mean():.4f}  miss mass/layer "
              f"{miss_w.mean():.4f}  misses/token {per_tok.mean():.1f}  host MB/token {mb:.0f}  worst layer hit "
              f"{routed_hit.mean((0, 2, 3)).min():.3f}")


if __name__ == "__main__":
    main()
