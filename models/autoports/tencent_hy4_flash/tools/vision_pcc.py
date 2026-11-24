# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device ViT + merger (tt/vision.py) vs the CPU reference (tools/vision_ref.py), and its latency. ViT only (no LM).

    ~/m27-work/hy4/run.sh python -m models.autoports.tencent_hy4_flash.tools.vision_pcc [--ref DIR/ref_512x512.pt ...]
        [--wdtype bfp8|bf16] [--layers 27] [--iters 5] [--sizes 512,1024]

Prints the PCC of every block's input (vs HF bf16), of the features (vs HF bf16 and fp32) and the warm latency of
one image (host preprocessing excluded) for each reference given and each extra --sizes image (latency only).
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import ttnn


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    d = os.path.expanduser("~/m27-work/hy4/vision")
    ap.add_argument("--ref", default=f"{d}/ref_512x512.pt,{d}/ref_1024x1024.pt")
    ap.add_argument("--wdtype", default=os.environ.get("HY4_VIT_WDTYPE", "bfp8"))
    ap.add_argument("--layers", type=int, default=27)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--states", action="store_true", help="per-block PCC (reads back every block input)")
    args = ap.parse_args()
    from models.autoports.tencent_hy4_flash.tools.generate import open_mesh
    from models.autoports.tencent_hy4_flash.tt.vision import Hy4Vision

    mesh = open_mesh()
    try:
        vit = Hy4Vision(mesh, wdtype=args.wdtype, layers=args.layers)
        for path in [p for p in args.ref.split(",") if p]:
            ref = torch.load(path, weights_only=False)
            pv, grid = ref["pixel_values"].float().numpy(), tuple(int(v) for v in ref["grid"][0].tolist())
            states = [] if args.states else None
            t0 = time.time()
            feat = vit.forward(pv, grid, states=states)
            t_first = time.time() - t0
            name = os.path.basename(path)
            if states is not None:
                for i, s in enumerate(states):
                    if i < len(ref["states"]) and (i % 3 == 0 or i >= args.layers - 1):
                        print(f"{name} block-in {i:2d}: PCC {pcc(s, ref['states'][i]):.6f}")
            if args.layers == 27:
                print(f"{name} features {tuple(feat.shape)}: PCC vs HF bf16 {pcc(feat, ref['feat']):.6f}"
                      + (f", vs fp32 {pcc(feat, ref['feat32']):.6f}" if "feat32" in ref else "")
                      + f"; HF bf16 vs fp32 {pcc(ref['feat'], ref['feat32']):.6f}" if "feat32" in ref else "")
                # per-token cosine spread (a few bad tokens would hide in a global PCC)
                a, b = feat.float(), ref.get("feat32", ref["feat"]).float()
                cs = torch.nn.functional.cosine_similarity(a, b, dim=-1)
                print(f"{name} per-token cosine min {float(cs.min()):.5f} mean {float(cs.mean()):.5f}")
            ts = []
            for _ in range(args.iters):
                ttnn.synchronize_device(mesh)
                t0 = time.time()
                vit.forward(pv, grid)
                ts.append(time.time() - t0)
            print(f"{name} grid {grid}: first {t_first * 1e3:.0f} ms, warm {np.median(ts) * 1e3:.1f} ms "
                  f"(min {min(ts) * 1e3:.1f}) [{args.wdtype}, {args.layers} blocks]")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
