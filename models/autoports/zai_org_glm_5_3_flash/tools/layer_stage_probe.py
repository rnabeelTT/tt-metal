# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic: one decoder layer's prefill stage by stage (mHC collapse, norm, attention, residual mix, FFN) vs the
HF layer's sub-block taps.

--hf32 <.pt> (tests/hf_ref/hf_fp32_taps.py on the same bf16 reference): every stage is also compared with the HF layer
run in fp32 (rms ratio TT / HF fp32), and the attention / FFN / routed / shared parts are run on HF's bf16 sub-block
inputs, so an update-scale offset against HF fp32 can be located to a sub-block (stage-1 review #5 P2-B).
--ref-n N: the bf16 reference has N >= n rows (segments [N]; e.g. the decode-stage probe's reference); rows [:n] are
used (causal)."""

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
    ap.add_argument("--ref-n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--hf32", default=None)
    ap.add_argument("--dump", default=None, help="save the TT FFN input / outputs (.pt) for host analysis")
    args = ap.parse_args()
    N = args.ref_n or args.n
    rd = CheckpointReader()
    weights = "real" if args.weights == "real" else {"synthetic": 0}
    if weights == "real":
        req = {"layer": args.layer, "weights": "real", "tokens": tokens(args.seed, N), "segments": [N], "taps": True}
    else:
        req = {
            "layer": args.layer,
            "weights": {"synthetic": 0, "stats": str(STATS)},
            "x": synthetic_stream(args.layer, 2, args.n),
            "segments": [args.n],
            "taps": True,
        }
    r = hf_layer_reference(req)
    taps = {k: v.float().reshape(N, -1)[: args.n] for k, v in r["taps"].items()}
    x = (r["x"] if "x" in r else req["x"]).float().reshape(N, H4)[: args.n]
    y_hf = r["y"].float().reshape(N, H4)[: args.n]
    t32 = {}
    if args.hf32:
        h32 = torch.load(args.hf32, weights_only=False)
        t32 = {k: v.float().reshape(N, -1)[: args.n] for k, v in h32["taps"].items()}
        t32["y"] = h32["y"].float().reshape(N, H4)[: args.n]
        for k in ("ffn_out_on_hf_in", "routed_on_hf_in", "shared_on_hf_in"):
            if k in h32:
                t32[k] = h32[k].float().reshape(N, -1)[: args.n]
    n = 128 if args.n <= 128 else 2048
    mesh = open_mesh()
    try:
        dec = build_layer(mesh, rd, args.layer, weights, batch=8, prefill_lengths=(128, 2048))
        xp = torch.zeros(1, 1, n, H4)
        xp[0, 0, : args.n] = x
        X = ttnn.from_torch(xp, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
        inputs = dec.prefill_inputs(n, start=0, length=args.n, slot=0)
        get = lambda t: ttnn.to_torch(t)[0, 0, : args.n].float()

        def rep(name, t, ref, ref32=None):
            s32 = ""
            if ref32 is not None:
                s32 = f" | vs HF fp32: PCC {pcc(t, ref32):.6f} rms ratio {(t.norm() / ref32.norm()).item():.4f}"
            print(f"{name:18s} PCC {pcc(t, ref):.6f} rms ratio {(t.norm() / ref.norm()).item():.4f}{s32}")

        h, post, comb = dec.attn_hc.pre(X)
        rep("attn_collapsed", get(h), taps["attn_collapsed"], t32.get("attn_collapsed"))
        xa = dec._norm(h, dec.input_norm_w)
        xa_dtype = xa.dtype
        rep("attn_in", get(xa), taps["attn_in"], t32.get("attn_in"))
        if dec.attn_kind == "kda":
            y, _ = dec.attn.prefill_forward(xa, dec.attn.new_prefill_state(), inputs["attn"])
        else:
            y, _ = dec.attn.prefill_forward(xa, inputs=inputs["attn"])
        rep("attn_out", get(y), taps["attn_out"], t32.get("attn_out"))
        X1 = dec.attn_hc.post(y, X, post, comb)
        X1h = get(X1)
        rep("X1", X1h, taps["X1"], t32.get("X1"))
        rep("X1 - X", X1h - x, taps["X1"] - x, t32["X1"] - x if t32 else None)
        h2, post2, comb2 = dec.ffn_hc.pre(X1)
        rep("ffn_collapsed", get(h2), taps["ffn_collapsed"], t32.get("ffn_collapsed"))
        xf = dec._norm(h2, dec.post_norm_w)
        rep("ffn_in", get(xf), taps["ffn_in"], t32.get("ffn_in"))
        xf_h = get(xf)
        z = dec.mlp(xf) if dec.mlp_kind == "dense" else dec.mlp.forward(xf, mode="prefill")
        z_h = get(z)
        rep("ffn_out", z_h, taps["ffn_out"], t32.get("ffn_out"))
        X2 = dec.ffn_hc.post(z, X1, post2, comb2)
        rep("X2 - X1", get(X2) - X1h, y_hf - taps["X1"], t32["y"] - t32["X1"] if t32 else None)
        rep("X2 - X (update)", get(X2) - x, y_hf - x, t32["y"] - x if t32 else None)
        if t32:
            rep("HF bf16 update", y_hf - x, y_hf - x, t32["y"] - x)
            rep("HF ffn_out bf16", taps["ffn_out"], taps["ffn_out"], t32["ffn_out_on_hf_in"])
        # attention on the HF attention input (isolates the attention from upstream error)
        xa_hf = ttnn.from_torch(
            torch.nn.functional.pad(taps["attn_in"], (0, 0, 0, n - args.n)).reshape(1, 1, n, -1),
            dtype=xa_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
        )
        inputs = dec.prefill_inputs(n, start=0, length=args.n, slot=1)
        if dec.attn_kind == "kda":
            y2, _ = dec.attn.prefill_forward(xa_hf, dec.attn.new_prefill_state(), inputs["attn"])
        else:
            y2, _ = dec.attn.prefill_forward(xa_hf, inputs=inputs["attn"])
        rep("attn_out(hf in)", get(y2), taps["attn_out"], t32.get("attn_out"))
        # FFN on the HF input (isolates the FFN from upstream error)
        xin = ttnn.from_torch(
            torch.nn.functional.pad(taps["ffn_in"], (0, 0, 0, n - args.n)).reshape(1, 1, n, -1).bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
        )
        z2 = dec.mlp(xin) if dec.mlp_kind == "dense" else dec.mlp.forward(xin, mode="prefill")
        rep("ffn_out(hf in)", get(z2), taps["ffn_out"], t32.get("ffn_out_on_hf_in"))
        if t32 and "routed_on_hf_in" in t32 and hasattr(dec.mlp, "_shared"):
            sh = get(dec.mlp._shared(xin))
            rep("shared(hf in)", sh, t32["shared_on_hf_in"], t32["shared_on_hf_in"])
            rep("routed(hf in)", get(z2) - sh, t32["routed_on_hf_in"], t32["routed_on_hf_in"])
        if args.dump:
            torch.save({"ffn_in": xf_h, "ffn_out": z_h, "ffn_out_on_hf_in": get(z2)}, args.dump)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
