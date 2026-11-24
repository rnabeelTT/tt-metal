# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Device teacher-forced log-probs vs the Hy4-flash CPU BF16 reference (ref_hy4.py tf: ref_<name>.npz).

usage: python compare_hy4.py <texts.json> <device out dir> <reference dir>

The device side writes, per sequence, either
  lp_<tag>_<name>.npy     full log-probs [T, V] (fp16 / fp32), as the M2.7 / Inkling tools do, or
  dev_<tag>_<name>.npz    `top1` [T] (argmax id) and `lp_at_ref` [T, 32] (its log-probs at the reference's top-32 ids,
                          in the reference's order), and optionally `nll` [T-1]
(<tag> is free text, e.g. the precision config; one row per (tag, name)).

Metrics per sequence and their mean per tag: top1_agree (device argmax == reference argmax), top1_agree_confident
(where the reference's top-1 probability > 0.5), dev_top1_in_ref_top5, kl_top32 (KL(ref || dev) on the reference's
top-32 support: truncated, slightly undercounts), kl_confident, ref_nll / dev_nll of the real next token.
"""

import glob
import json
import os
import sys

import numpy as np


def load_dev(path, ref_ids, T, nxt):
    if path.endswith(".npy"):
        lp = np.load(path).astype(np.float32)[:T]
        return lp.argmax(-1), np.take_along_axis(lp, ref_ids, 1), -lp[np.arange(len(nxt)), nxt]
    d = np.load(path)
    nll = d["nll"][: len(nxt)] if "nll" in d else None
    return d["top1"][:T], d["lp_at_ref"][:T].astype(np.float32), nll


def main():
    texts, dev_dir, ref_dir = sys.argv[1:4]
    seqs = json.load(open(texts))
    rows = []
    paths = sorted(glob.glob(os.path.join(dev_dir, "lp_*_*.npy")) + glob.glob(os.path.join(dev_dir, "dev_*_*.npz")))
    for path in paths:
        base = os.path.basename(path).split(".")[0]
        _, rest = base.split("_", 1)
        name = next((n for n in sorted(seqs, key=len, reverse=True) if rest.endswith("_" + n)), None)
        if name is None:
            continue
        tag = rest[: -len(name) - 1]
        ref = np.load(os.path.join(ref_dir, f"ref_{name}.npz"))
        ids, lp = ref["top_ids"].astype(np.int64), ref["top_lp"].astype(np.float32)
        T = len(seqs[name]) - 1
        nxt = np.array(seqs[name][1:])
        top1, lp_dev, dev_nll = load_dev(path, ids[:T], T, nxt)
        p_ref = np.exp(lp[:T])
        kl = (p_ref * (lp[:T] - lp_dev)).sum(-1)
        conf = p_ref[:, 0] > 0.5
        agree = top1 == ids[:T, 0]
        r = {
            "tag": tag,
            "name": name,
            "top1_agree": round(float(agree.mean()), 4),
            "top1_agree_confident": round(float(agree[conf].mean()), 4),
            "dev_top1_in_ref_top5": round(float((ids[:T, :5] == top1[:, None]).any(-1).mean()), 4),
            "kl_top32": round(float(kl.mean()), 4),
            "kl_confident": round(float(kl[conf].mean()), 4),
            "ref_nll": round(float(ref["nll"].mean()), 4),
            "dev_nll": None if dev_nll is None else round(float(np.mean(dev_nll)), 4),
        }
        rows.append(r)
        print(json.dumps(r))
    for tag in sorted({r["tag"] for r in rows}):
        rs = [r for r in rows if r["tag"] == tag]
        print(
            f"MEAN {tag} (n={len(rs)}): "
            + " ".join(
                f"{k} {np.mean([r[k] for r in rs]):.4f}"
                for k in ("top1_agree", "top1_agree_confident", "kl_top32", "kl_confident")
            )
        )
    json.dump(rows, open(os.path.join(dev_dir, "compare.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
