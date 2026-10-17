# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic (stage-1 review #3 P2): one decoder layer's DECODE step, sub-block by sub-block, vs the HF layer's taps.

HF runs the real-token sequence (prefill n + D tokens) in one pass with sub-block taps; row n + i of every tap is the
reference of decode step i (causal). TT prefills n tokens into slot 5 of an 8-slot layer, then decodes D tokens with
decode_forward split into its stages. Per stage and step: PCC and RMS ratio TT / HF of that token's row, plus the FFN
on the HF FFN input (isolates the FFN from upstream error) and the router margin of the token.

--hf32 <.pt> (tests/hf_ref/hf_fp32_taps.py on the same reference): the main stages are also compared with the HF layer
run in fp32 ("... vs fp32" rows; stage-1 review #5 P2-B).

usage: python -m models.autoports.zai_org_glm_5_3_flash.tools.decode_stage_probe --layer 4 --n 100 --steps 12"""

import argparse

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.harness import (
    H4,
    DecodeRunner,
    build_layer,
    open_mesh,
    pcc,
    prefill_user,
)
from models.autoports.zai_org_glm_5_3_flash.tests.hf_ref.client import hf_layer_reference
from models.autoports.zai_org_glm_5_3_flash.tests.test_functional_decoder import tokens
from models.autoports.zai_org_glm_5_3_flash.tt.ckpt import CheckpointReader
from models.demos.deepseek_v3_d_p.reference.mhc.mhc_reference import MHCConfig, parametrize

SLOT, BATCH = 5, 8
HC = 4


def hc_params(hc, sd_fn, sd_base, sd_scale, X, tf32=False):
    """Host mHC (post, comb) of stream row X [n*C] in fp32 (HF math); tf32: X rounded to TF32 first (the matmul unit's
    reading of fp32 operands)."""
    cfg = hc.cfg
    X = X.float()
    Xm = (X.view(torch.int32) & ~0x1FFF).view(torch.float32) if tf32 else X
    fn = sd_fn.to(torch.bfloat16).float()
    mixes = (Xm @ fn.T) * torch.rsqrt((X * X).mean() + cfg.norm_eps)
    mc = MHCConfig(dim=X.numel() // HC, n=HC, sinkhorn_iters=cfg.sinkhorn_iters, eps=cfg.eps, norm_eps=cfg.norm_eps)
    _, post, comb = parametrize(mixes, sd_scale.to(torch.bfloat16).float(), sd_base.to(torch.bfloat16).float(), mc)
    return post, comb.reshape(-1)


def hc_post(yv, resid, post, comb):
    """streams'[i] = post[i] y + sum_j comb[j, i] streams[j] (fp32; mhc_post semantics)."""
    S = resid.reshape(HC, -1).float()
    c = comb[: HC * HC].reshape(HC, HC).float()
    return (post[:HC, None].float() * yv.float()[None] + c.T @ S).reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--hf32", default=None)
    a = ap.parse_args()
    N = a.n + a.steps
    r = hf_layer_reference(
        {"layer": a.layer, "weights": "real", "tokens": tokens(a.seed, N), "segments": [N], "taps": True}
    )
    taps = {k: v.float().reshape(N, -1) for k, v in r["taps"].items() if torch.is_tensor(v)}
    x = r["x"].float().reshape(N, H4)
    y = r["y"].float().reshape(N, H4)
    margin = r.get("route_margin")
    margin = None if margin is None else margin.reshape(-1).float()
    t32 = None
    if a.hf32:
        h32 = torch.load(a.hf32, weights_only=False)
        t32 = {k: v.float().reshape(N, -1) for k, v in h32["taps"].items()}
        t32["y"] = h32["y"].float().reshape(N, H4)
        t32["ffn_out_on_hf_in"] = h32["ffn_out_on_hf_in"].float().reshape(N, -1)
    rd = CheckpointReader()
    mesh = open_mesh()
    stats = {}
    try:
        dec = build_layer(mesh, rd, a.layer, "real", batch=BATCH, prefill_lengths=(128, 2048))
        sd = rd.layer_state(a.layer)
        hcw = {k: (sd[f"hc_{k}_fn"], sd[f"hc_{k}_base"], sd[f"hc_{k}_scale"]) for k in ("attn", "ffn")}
        prefill_user(dec, mesh, x, a.n, SLOT)
        run = DecodeRunner(dec, mesh, BATCH)
        row = lambda t: ttnn.to_torch(t)[0, 0, SLOT].float()
        for i in range(a.steps):
            p = a.n + i
            pos = [-1] * BATCH
            pos[SLOT] = p
            run._set_inputs({SLOT: x[p]}, pos)
            X = run.x_buf
            R = {}  # rows read at once: later stages may free their inputs
            h, post, comb = dec.attn_hc.pre(X)
            R["h"], R["post"], R["comb"] = row(h), row(post), row(comb)
            xa = dec._norm(h, dec.input_norm_w)
            R["xa"] = row(xa)
            if dec.attn_kind == "kda":
                ya = dec.attn.decode_forward(xa, dec.decode_state, inputs=run.dsa_bufs)
            else:
                ya = dec.attn.decode_forward(xa, inputs=run.dsa_bufs)
            R["ya"] = row(ya)
            X1 = dec.attn_hc.post(ya, X, post, comb)
            R["X1"] = row(X1)
            h2, post2, comb2 = dec.ffn_hc.pre(X1)
            R["h2"], R["post2"], R["comb2"] = row(h2), row(post2), row(comb2)
            xf = dec._norm(h2, dec.post_norm_w)
            R["xf"] = row(xf)
            z = dec.mlp(xf) if dec.mlp_kind == "dense" else dec.mlp.forward(xf, mode="decode")
            R["z"] = row(z)
            X2 = dec.ffn_hc.post(z, X1, post2, comb2)
            R["X2"] = row(X2)
            fin = torch.zeros(1, 1, BATCH, taps["ffn_in"].shape[-1])
            fin[0, 0, SLOT] = taps["ffn_in"][p]
            tf = ttnn.from_torch(fin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
            z_hf = dec.mlp(tf) if dec.mlp_kind == "dense" else dec.mlp.forward(tf, mode="decode")
            R["z_hf"] = row(z_hf)
            got = {
                "attn_collapsed": (R["h"], taps["attn_collapsed"][p]),
                "attn_in": (R["xa"], taps["attn_in"][p]),
                "attn_out": (R["ya"], taps["attn_out"][p]),
                "X1 - X": (R["X1"] - x[p], taps["X1"][p] - x[p]),
                "ffn_collapsed": (R["h2"], taps["ffn_collapsed"][p]),
                "ffn_in": (R["xf"], taps["ffn_in"][p]),
                "ffn_out": (R["z"], taps["ffn_out"][p]),
                "ffn_out(hf in)": (R["z_hf"], taps["ffn_out"][p]),
                "X2 - X1": (R["X2"] - R["X1"], y[p] - taps["X1"][p]),
                "X2 - X": (R["X2"] - x[p], y[p] - x[p]),
                # host mHC post on TT post / comb: with the TT sub-block output (checks the kernel and the orientation),
                # then with the HF sub-block output (an excess that remains comes from post / comb, not from y)
                "X1-X host tt y": (hc_post(R["ya"], x[p], R["post"], R["comb"]) - x[p], taps["X1"][p] - x[p]),
                "X1-X host hf y": (
                    hc_post(taps["attn_out"][p], x[p], R["post"], R["comb"]) - x[p],
                    taps["X1"][p] - x[p],
                ),
                "X1-X host fp32 pc": (
                    hc_post(taps["attn_out"][p], x[p], *hc_params(dec.attn_hc, *hcw["attn"], x[p])) - x[p],
                    taps["X1"][p] - x[p],
                ),
                "X1-X host tf32 pc": (
                    hc_post(taps["attn_out"][p], x[p], *hc_params(dec.attn_hc, *hcw["attn"], x[p], tf32=True)) - x[p],
                    taps["X1"][p] - x[p],
                ),
                "X2-X1 hostfp32 pc": (
                    hc_post(taps["ffn_out"][p], taps["X1"][p], *hc_params(dec.ffn_hc, *hcw["ffn"], taps["X1"][p]))
                    - taps["X1"][p],
                    y[p] - taps["X1"][p],
                ),
                "X2-X1 host hf y": (
                    hc_post(taps["ffn_out"][p], R["X1"], R["post2"], R["comb2"]) - R["X1"],
                    y[p] - taps["X1"][p],
                ),
            }
            if t32 is not None:
                got.update(
                    {
                        "attn_out vs fp32": (R["ya"], t32["attn_out"][p]),
                        "X1 - X vs fp32": (R["X1"] - x[p], t32["X1"][p] - x[p]),
                        "ffn_out vs fp32": (R["z"], t32["ffn_out"][p]),
                        "ffn(hf in) vs fp32": (R["z_hf"], t32["ffn_out_on_hf_in"][p]),
                        "X2 - X1 vs fp32": (R["X2"] - R["X1"], t32["y"][p] - t32["X1"][p]),
                        "X2 - X vs fp32": (R["X2"] - x[p], t32["y"][p] - x[p]),
                        "HF16 upd vs fp32": (y[p] - x[p], t32["y"][p] - x[p]),
                    }
                )
            m = "" if margin is None else f" margin {float(margin[p]):.1e}"
            print(f"--- step {i} pos {p}{m}")
            for k, (t, ref) in got.items():
                ratio = (t.norm() / ref.norm()).item()
                stats.setdefault(k, []).append(ratio)
                print(f"{k:16s} PCC {pcc(t, ref):.6f}  rms ratio {ratio:.4f}")
        print("=== mean rms ratio over steps")
        for k, v in stats.items():
            v = torch.tensor(v)
            print(f"{k:16s} mean {v.mean():.4f} min {v.min():.4f} max {v.max():.4f}")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
