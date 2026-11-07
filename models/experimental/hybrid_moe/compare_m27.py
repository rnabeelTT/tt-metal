# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Device teacher-forced log-probs (run_generate.py POC_TF_TEXTS: lp_<tau>_<name>.npy) vs the MiniMax-M2.7 CPU
reference (ref_m27.py: ref_<name>.npz with the reference's top-32 log-probs per position).

usage: python compare_m27.py <texts.json> <device out dir> <reference dir>
KL is truncated to the reference's top-32 support (the reference keeps only those), so it slightly undercounts.
"""

import glob
import json
import os
import sys

import numpy as np


def main():
    texts, dev_dir, ref_dir = sys.argv[1:4]
    seqs = json.load(open(texts))
    rows = []
    for path in sorted(glob.glob(os.path.join(dev_dir, "lp_*_*.npy"))):
        tau, name = os.path.basename(path)[3:-4].split("_", 1)
        if name not in seqs:
            continue
        hyb = np.load(path).astype(np.float32)
        ref = np.load(os.path.join(ref_dir, f"ref_{name}.npz"))
        ids, lp = ref["top_ids"].astype(np.int64), ref["top_lp"].astype(np.float32)
        T = len(seqs[name]) - 1
        nxt = np.array(seqs[name][1:])
        h_top1 = hyb[:T].argmax(-1)
        p_ref = np.exp(lp[:T])
        kl = (p_ref * (lp[:T] - np.take_along_axis(hyb[:T], ids[:T], 1))).sum(-1)
        conf = p_ref[:, 0] > 0.5
        r = {
            "name": name,
            "tau": float(tau),
            "top1_agree": round(float((h_top1 == ids[:T, 0]).mean()), 4),
            "top1_agree_confident": round(float((h_top1 == ids[:T, 0])[conf].mean()), 4),
            "hyb_top1_in_ref_top5": round(float((ids[:T, :5] == h_top1[:, None]).any(-1).mean()), 4),
            "kl_top32": round(float(kl.mean()), 4),
            "kl_confident": round(float(kl[conf].mean()), 4),
            "ref_nll": round(float(ref["nll"].mean()), 4),
            "hyb_nll": round(float(-hyb[np.arange(T), nxt].mean()), 4),
        }
        rows.append(r)
        print(json.dumps(r))
    for tau in sorted({r["tau"] for r in rows}):
        rs = [r for r in rows if r["tau"] == tau]
        print(
            f"MEAN tau={tau}: "
            + " ".join(
                f"{k} {np.mean([r[k] for r in rs]):.4f}"
                for k in ("top1_agree", "top1_agree_confident", "kl_top32", "kl_confident")
            )
        )
    json.dump(rows, open(os.path.join(dev_dir, "compare.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
