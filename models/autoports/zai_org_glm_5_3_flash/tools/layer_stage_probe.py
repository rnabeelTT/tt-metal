# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic: one decoder layer's prefill stage by stage (mHC collapse, norm, attention, residual mix, FFN) vs the
HF layer's sub-block taps."""

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
    args = ap.parse_args()
    rd = CheckpointReader()
    weights = "real" if args.weights == "real" else {"synthetic": 0}
    if weights == "real":
        req = {"layer": args.layer, "weights": "real", "tokens": tokens(1, args.n), "segments": [args.n], "taps": True}
    else:
        req = {
            "layer": args.layer,
            "weights": {"synthetic": 0, "stats": str(STATS)},
            "x": synthetic_stream(args.layer, 2, args.n),
            "segments": [args.n],
            "taps": True,
        }
    r = hf_layer_reference(req)
    taps = {k: v.float().reshape(args.n, -1) for k, v in r["taps"].items()}
    x = (r["x"] if "x" in r else req["x"]).float().reshape(args.n, H4)
    n = 128 if args.n <= 128 else 2048
    mesh = open_mesh()
    try:
        dec = build_layer(mesh, rd, args.layer, weights, batch=8, prefill_lengths=(128, 2048))
        xp = torch.zeros(1, 1, n, H4)
        xp[0, 0, : args.n] = x
        X = ttnn.from_torch(xp, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
        inputs = dec.prefill_inputs(n, start=0, length=args.n, slot=0)
        get = lambda t: ttnn.to_torch(t)[0, 0, : args.n].float()

        def rep(name, t, ref):
            print(
                f"{name:16s} PCC {pcc(t, ref):.6f}  rms tt {t.pow(2).mean().sqrt():.4e} ref {ref.pow(2).mean().sqrt():.4e}"
            )

        h, post, comb = dec.attn_hc.pre(X)
        rep("attn_collapsed", get(h), taps["attn_collapsed"])
        xa = dec._norm(h, dec.input_norm_w)
        rep("attn_in", get(xa), taps["attn_in"])
        if dec.attn_kind == "kda":
            y, _ = dec.attn.prefill_forward(xa, dec.attn.new_prefill_state(), inputs["attn"])
        else:
            y, _ = dec.attn.prefill_forward(xa, inputs=inputs["attn"])
        rep("attn_out", get(y), taps["attn_out"])
        X1 = dec.attn_hc.post(y, X, post, comb)
        rep("X1", get(X1), taps["X1"])
        rep("X1 - X", get(X1) - x, taps["X1"] - x)
        h2, post2, comb2 = dec.ffn_hc.pre(X1)
        rep("ffn_collapsed", get(h2), taps["ffn_collapsed"])
        xf = dec._norm(h2, dec.post_norm_w)
        rep("ffn_in", get(xf), taps["ffn_in"])
        z = dec.mlp(xf) if dec.mlp_kind == "dense" else dec.mlp.forward(xf, mode="prefill")
        rep("ffn_out", get(z), taps["ffn_out"])
        # FFN on the HF input (isolates the FFN from upstream error)
        xin = ttnn.from_torch(
            torch.nn.functional.pad(taps["ffn_in"], (0, 0, 0, n - args.n)).reshape(1, 1, n, -1).bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
        )
        z2 = dec.mlp(xin) if dec.mlp_kind == "dense" else dec.mlp.forward(xin, mode="prefill")
        rep("ffn_out(hf in)", get(z2), taps["ffn_out"])
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
