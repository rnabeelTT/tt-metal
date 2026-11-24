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
    ap.add_argument("--decode-from", type=int, default=0,
                    help="P > 0: prefill the first P tokens, then feed the rest through the traced decoder (tt/decode.py) "
                         "one row per pass: gates the decode path")
    ap.add_argument("--rows", type=int, default=1, help="rows per decoder pass (--decode-from)")
    ap.add_argument("--slot", type=int, default=0, help="run in this sequence slot (cache region u x rows)")
    args = ap.parse_args()
    if args.slot:
        os.environ["HY4_SLOTS"] = str(max(int(os.environ.get("HY4_SLOTS", "1")), args.slot + 1))
    from models.autoports.tencent_hy4_flash.tools.generate import build, open_mesh

    seqs = json.load(open(args.texts))
    names = [n for n in (args.names.split(",") if args.names else sorted(seqs))]
    os.makedirs(args.out_dir, exist_ok=True)
    mesh = open_mesh()
    try:
        model = build(mesh, args)
        if model.engram is not None and hasattr(model.engram.store, "wait_ram"):
            model.engram.store.wait_ram()
        decs = {}
        if args.slot:
            model.set_slot(args.slot)
        for name in names:
            ids = torch.tensor(seqs[name], dtype=torch.long)
            ref = np.load(os.path.join(args.ref_dir, f"ref_{name}.npz"))
            top_ids = ref["top_ids"].astype(np.int64)
            model.reset()
            t0 = time.time()
            lps = []
            P = args.decode_from if 0 < args.decode_from < ids.numel() else ids.numel()
            for c0 in range(0, P, args.chunk):
                lg = model.forward(ids[c0 : min(P, c0 + args.chunk)], logits_rows="all")
                lps.append(torch.log_softmax(lg.float(), dim=-1))
            if P < ids.numel():
                from models.autoports.tencent_hy4_flash.tt.decode import Hy4Decoder

                dec = decs.get("d")
                if dec is None:
                    dec = decs["d"] = Hy4Decoder(model, trace=True)
                dec.start(P)
                dec.set_history(ids[:P].tolist())
                for i in range(P, ids.numel(), args.rows):
                    n = min(args.rows, ids.numel() - i)
                    lg = dec.run([int(t) for t in ids[i : i + n]], logits_rows="all", p0=i)
                    lps.append(torch.log_softmax(lg.float().reshape(n, -1), dim=-1))
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
