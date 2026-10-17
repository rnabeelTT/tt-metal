# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HF router near-tie counterfactual for one GLM-5.3-Flash MoE decoder layer (runs in the HF venv, never imports ttnn).

For the requested rows, the unmodified HF layer runs again on the cached reference input with the router's 8th expert
replaced by its 9th (selection score sigmoid(logit) + correction bias; the routing weights renormalized as HF does,
times routed_scaling_factor). Every other row keeps HF's own routing and must reproduce the cached output bit-exactly.
A decoded token whose 8th / 9th selection margin is small can route that one expert differently on the port (its FFN
input differs from HF's by rounding): such a token is compared with y_swap as well as with y
(tests/test_functional_decoder.py, check()).

The MoE output of a row does not feed the layer's caches (the KDA state and the DSA latent / indexer rows come from the
attention input), so all requested rows are swapped in one pass.

Request (torch.save dict, --request): {"flip": base reference key, "layer", "weights": "real", "segments",
"rows": [row indices]}; --base: the cached base reference result (hf_layer_ref.py output with "x" and "y").
Result: rows [R] int64, y_swap [R, 4, 4096] bf16, experts [R, 2] int64 (HF's 8th, the 9th that replaces it),
exact (max |y - y_ref| over the other rows), seconds.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("glm53_hf_layer_ref", HERE / "hf_layer_ref.py")
H = importlib.util.module_from_spec(_spec)
sys.modules["glm53_hf_layer_ref"] = H
_spec.loader.exec_module(H)

from transformers.cache_utils import DynamicCache  # noqa: E402


def swap_hook(gate, rows: set, experts: dict):
    """Forward hook on the router: for rows in `rows` (absolute row index over the run's segments), replace the 8th
    selected expert by the 9th and renormalize the weights as the router does."""
    state = {"offset": 0}
    k = gate.top_k

    def hook(m, args, out):
        logits, weights, idx = out
        n = logits.shape[0]
        local = [r - state["offset"] for r in sorted(rows) if state["offset"] <= r < state["offset"] + n]
        state["offset"] += n
        if not local:
            return out
        scores = logits.float().sigmoid()
        choice = scores + m.e_score_correction_bias
        assert m.num_group == 1, "group-limited routing not handled"
        top = torch.topk(choice, k=k + 1, dim=-1)  # sorted
        idx, weights = idx.clone(), weights.clone()
        for i in local:
            sel = torch.cat([top.indices[i, : k - 1], top.indices[i, k:]])
            assert set(sel[: k - 1].tolist()) <= set(idx[i].tolist()), "HF top-7 differs from the sorted top-k"
            w = scores[i].gather(0, sel)
            if m.norm_topk_prob:
                w = w / (w.sum() + 1e-20)
            idx[i] = sel
            weights[i] = (w * m.routed_scaling_factor).to(weights.dtype)
            experts[i + state["offset"] - n] = (int(top.indices[i, k - 1]), int(top.indices[i, k]))
        return logits, weights, idx

    return gate.register_forward_hook(hook)


@torch.no_grad()
def run(req: dict, base: dict) -> dict:
    t0 = time.time()
    ckpt_dir = Path(req.get("ckpt") or os.environ["GLM53_CKPT"])
    rd = H.ckpt.CheckpointReader(ckpt_dir)
    tc = H.text_config(ckpt_dir, rd.num_experts)
    layer = int(req["layer"])
    mod = H.build_layer(tc, layer, H.layer_weights(req["weights"], layer, rd))
    rows = sorted(int(r) for r in req["rows"])
    experts = {}
    swap_hook(mod.mlp.gate, set(rows), experts)
    x = base["x"].to(torch.bfloat16)
    B, N = x.shape[:2]
    assert B == 1, "one sequence per reference"
    segments = list(req["segments"])
    assert sum(segments) == N, (segments, N)
    cache = DynamicCache(config=tc)
    ys, s0 = [], 0
    for seg in segments:
        y, _ = mod(
            x[:, s0 : s0 + seg],
            attention_mask=torch.ones(B, seg, dtype=torch.bool),
            past_key_values=cache,
            use_cache=True,
            position_embeddings=None,
        )
        ys.append(y)
        s0 += seg
    y = torch.cat(ys, dim=1)
    keep = torch.ones(N, dtype=torch.bool)
    keep[rows] = False
    exact = float((y[0, keep].float() - base["y"][0, keep].float()).abs().max()) if keep.any() else 0.0
    return {
        "rows": torch.tensor(rows, dtype=torch.int64),
        "y_swap": y[0, rows].clone(),
        "experts": torch.tensor([experts[r] for r in rows], dtype=torch.int64).reshape(-1, 2),
        "exact": exact,
        "seconds": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    req = torch.load(args.request, weights_only=False)
    base = torch.load(args.base, weights_only=False, map_location="cpu")
    res = run(req, base)
    if res["exact"] != 0.0:
        raise SystemExit(f"rerun does not reproduce the base reference: max |y - y_ref| {res['exact']:.3e}")
    tmp = args.out + ".tmp"
    torch.save(res, tmp)
    os.replace(tmp, args.out)
    print(f"hf_flip_ref layer {req['layer']}: {len(res['rows'])} rows swapped in {res['seconds']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
