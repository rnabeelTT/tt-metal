# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Static-plan quality on recorded 397B routing with scores (run_generate HYB_RECORD: p<i> ids, s<i> scores).
Metric: computed misses per token (misses with routing weight >= tau, the host work under HYB_SKIP=tau) and the hit
rate, for a plan built from the TRAIN files and evaluated on the EVAL files.
usage: python sim_plan.py --train a.npz [...] --eval b.npz [...] [--C 255] [--tau 0.1]"""
import argparse

import numpy as np

E = 512


def load(paths):
    out = []
    for p in paths:
        z = np.load(p)
        for k in z.files:
            if k.startswith("p"):
                s = z["s" + k[1:]].astype(np.float32) if ("s" + k[1:]) in z.files else None
                out.append((z[k], s))
    return out


def freq(data, L, tau):
    f = np.zeros((L, E))
    for ids, sc in data:
        w = np.ones(ids.shape) if (tau <= 0 or sc is None) else (sc >= tau).astype(np.float64)
        for li in range(L):
            np.add.at(f[li], ids[:, li, :].ravel(), w[:, li, :].ravel())
    return f


def evaluate(data, f, C, tau):
    L = f.shape[0]
    res = np.zeros((L, E), bool)
    for li in range(L):
        res[li, np.argsort(-f[li], kind="stable")[:C]] = True
    picks = hits = computed = toks = 0
    for ids, sc in data:
        T = ids.shape[0]
        hit = res[np.arange(L)[None, :, None], ids]
        picks += hit.size
        hits += hit.sum()
        miss = ~hit
        computed += (miss & (sc >= tau)).sum() if sc is not None else miss.sum()
        toks += T
    return hits / picks, computed / toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--C", type=int, default=255)
    ap.add_argument("--tau", type=float, default=0.1)
    a = ap.parse_args()
    tr, ev = load(a.train), load(a.eval)
    L = tr[0][0].shape[1]
    for name, w in (("count", 0.0), (f"weight>={a.tau}", a.tau)):
        hr, cm = evaluate(ev, freq(tr, L, w), a.C, a.tau)
        print(f"plan {name:14s} C={a.C}: eval hit rate {hr:.3f}, computed misses/token {cm:.1f} "
              f"(per layer {cm / L:.2f})")


if __name__ == "__main__":
    main()
