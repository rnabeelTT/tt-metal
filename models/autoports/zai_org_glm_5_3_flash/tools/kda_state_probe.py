# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic: KDA prefill carry (recurrent state) vs HF, and the decay statistics of the layer's input."""

import argparse

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.harness import H4, STATS, build_layer, open_mesh, pcc
from models.autoports.zai_org_glm_5_3_flash.tests.hf_ref.client import hf_layer_reference
from models.autoports.zai_org_glm_5_3_flash.tests.test_functional_decoder import synthetic_stream, tokens
from models.autoports.zai_org_glm_5_3_flash.tt.ckpt import CheckpointReader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--weights", default="synthetic")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--bucket", type=int, default=2048)
    args = ap.parse_args()
    rd = CheckpointReader()
    weights = "real" if args.weights == "real" else {"synthetic": 0}
    if weights == "real":
        req = {
            "layer": args.layer,
            "weights": "real",
            "tokens": tokens(1, args.n),
            "segments": [args.n],
            "capture": True,
        }
    else:
        req = {
            "layer": args.layer,
            "weights": {"synthetic": 0, "stats": str(STATS)},
            "x": synthetic_stream(args.layer, 2, args.n),
            "segments": [args.n],
            "capture": True,
        }
    r = hf_layer_reference(req)
    x = (r["x"] if "x" in r else req["x"]).float().reshape(args.n, H4)
    ref_state = r["states"][0]["recurrent_state"].float()  # [1,64,128,128]
    mesh = open_mesh()
    try:
        dec = build_layer(mesh, rd, args.layer, weights, batch=8, prefill_lengths=(128, 2048))
        n = next(b for b in (128, 2048) if b >= args.n)
        xp = torch.zeros(1, 1, n, H4)
        xp[0, 0, : args.n] = x
        X = ttnn.from_torch(xp, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
        Y, st = dec.prefill_forward(X, inputs=dec.prefill_inputs(n, start=0, length=args.n, slot=0))
        s = ttnn.to_torch(st.recurrent).float()
        print("state PCC", pcc(s, ref_state), "max|ref|", ref_state.abs().max().item(), "max|tt|", s.abs().max().item())
        per_head = [pcc(s[0, h], ref_state[0, h]) for h in range(64)]
        print(
            "worst heads",
            sorted(range(64), key=lambda h: per_head[h])[:6],
            [round(per_head[h], 4) for h in sorted(range(64), key=lambda h: per_head[h])[:6]],
        )
        yt = ttnn.to_torch(Y)[0, 0, : args.n]
        y = r["y"].float().reshape(args.n, H4)
        print("layer delta PCC", pcc(yt - x, y - x))
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
