# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Agreement between two CPU references (ref_inkling.py output dirs), with the device eval's metrics: the noise floor
of the teacher-forced gate. Two equally valid implementations (e.g. REF_ROUTER=fp32 vs hf: the router in fp32 as vLLM
computes it vs transformers' all-bf16) disagree on near-ties; the device cannot be expected to agree with one reference
better than another valid implementation does.

top1 / top1_conf: argmax agreement (conf: positions where reference A is > 0.5 sure); kl32: KL(A || B) truncated to A's
top 32, B's log-probs at those ids taken from B's own top 32 (ids outside B's top 32 get B's 32nd log-prob: an upper
bound of B's probability there, so this kl32 is a slight underestimate).

usage: python compare_refs.py <ref dir A> <ref dir B> [texts.json]
"""

import glob
import os
import sys

import numpy as np


def main():
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    rows = []
    for fa in sorted(glob.glob(os.path.join(a_dir, "ref_*.npz"))):
        name = os.path.basename(fa)[4:-4]
        fb = os.path.join(b_dir, f"ref_{name}.npz")
        if not os.path.exists(fb):
            continue
        A, B = np.load(fa), np.load(fb)
        ai, al = A["top_ids"], A["top_lp"].astype(np.float64)
        bi, bl = B["top_ids"], B["top_lp"].astype(np.float64)
        T = min(len(ai), len(bi)) - 1
        top1 = ai[:T, 0] == bi[:T, 0]
        conf = np.exp(al[:T, 0]) > 0.5
        kl = np.zeros(T)
        for t in range(T):
            lut = dict(zip(bi[t].tolist(), bl[t].tolist()))
            floor = bl[t, -1]
            blt = np.array([lut.get(int(i), floor) for i in ai[t]])
            kl[t] = (np.exp(al[t]) * (al[t] - blt)).sum()
        rows.append((name, top1.mean(), top1[conf].mean(), kl.mean()))
        print(f"{name}: top1 {top1.mean():.4f}  top1_conf {top1[conf].mean():.4f}  kl32 {kl.mean():.4f}")
    m = np.mean([r[1:] for r in rows], 0)
    print(f"MEAN top1 {m[0]:.4f}  top1_conf {m[1]:.4f}  kl32 {m[2]:.4f}")


if __name__ == "__main__":
    main()
