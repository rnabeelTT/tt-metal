# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced device log-probs for the quality gate (models/experimental/hybrid_moe/compare_hy4.py).

    python -m models.autoports.tencent_hy4_flash.tools.teacher_force TEXTS.json REF_DIR OUT_DIR TAG [--names a,b]

TEXTS.json: {name: [token ids]} (the sequences ref_hy4 scored). For each sequence (a fresh model sequence, prefilled in
chunks) writes OUT_DIR/dev_<TAG>_<name>.npz: top1 [T], lp_at_ref [T, 32] (log-probs at the reference's top-32 ids)
and nll [T-1] of the real next tokens. Then run compare_hy4.py TEXTS.json OUT_DIR REF_DIR.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import ttnn
from loguru import logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("texts")
    ap.add_argument("ref_dir")
    ap.add_argument("out_dir")
    ap.add_argument("tag")
    ap.add_argument("--names", default="")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--layers", default="")
    ap.add_argument("--no-engram", action="store_true")
    args = ap.parse_args()
    from models.autoports.tencent_hy4_flash.tools.generate import build, open_mesh

    seqs = json.load(open(args.texts))
    names = [n for n in (args.names.split(",") if args.names else sorted(seqs))]
    os.makedirs(args.out_dir, exist_ok=True)
    mesh = open_mesh()
    try:
        model = build(mesh, args)
        for name in names:
            ids = torch.tensor(seqs[name], dtype=torch.long)
            ref = np.load(os.path.join(args.ref_dir, f"ref_{name}.npz"))
            top_ids = ref["top_ids"].astype(np.int64)
            model.reset()
            t0 = time.time()
            lps = []
            for c0 in range(0, ids.numel(), args.chunk):
                lg = model.forward(ids[c0 : c0 + args.chunk], logits_rows="all")
                lps.append(torch.log_softmax(lg.float(), dim=-1))
            lp = torch.cat(lps)  # [T, V]
            T = ids.numel() - 1
            lp = lp[:T]
            nxt = ids[1:]
            nll = (-lp[torch.arange(T), nxt]).numpy()
            np.savez(
                os.path.join(args.out_dir, f"dev_{args.tag}_{name}.npz"),
                top1=lp.argmax(-1).numpy(),
                lp_at_ref=torch.gather(lp, 1, torch.from_numpy(top_ids[:T])).numpy(),
                nll=nll,
            )
            logger.info(f"{name}: {ids.numel()} tokens in {time.time() - t0:.1f} s, nll {nll.mean():.4f}")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
