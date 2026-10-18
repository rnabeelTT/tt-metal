# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Quantization-aware HF reference for one GLM-5.3-Flash decoder layer (runs in the HF venv, never imports ttnn).

The unmodified HF layer (hf_layer_ref.py) with selected weights rounded exactly as the port stores them on device
(bfp.py: the tt-metal host block-float packing, checked bit-exactly by tools/bfp_check.py). A precision policy that
stores weights below the checkpoint precision (bfp4 routed experts) is then held to the functional bars against its
own quantized model; the checkpoint reference stays the measure of the quantization loss itself.

Request (torch.save dict, --request):
  quant:    {fnmatch pattern over checkpoint names: "bfp4" | "bfp8"} (e.g. {"mlp.experts.*": "bfp4"})
  layer, weights, segments, taps, mask: as hf_layer_ref.py
  x:        the input stream (the base reference's recorded x for token requests: the stream does not depend on this
            layer's weights)
  rows:     optional: the router near-tie counterfactual of hf_flip_ref.py on these rows (8th expert -> 9th), with the
            quantized weights; the result is then {rows, y_swap, experts} as hf_flip_ref.py
Result: as hf_layer_ref.py (y, route_margin, taps), or the y_swap dict.
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


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bfp = _load("glm53_bfp", HERE / "bfp.py")
F = _load("glm53_hf_flip_ref", HERE / "hf_flip_ref.py")  # loads hf_layer_ref.py as F.H
H = F.H


@torch.no_grad()
def run(req: dict) -> dict:
    t0 = time.time()
    spec = dict(req["quant"])
    rows = sorted(int(r) for r in req.get("rows") or [])
    orig_weights, orig_build = H.layer_weights, H.build_layer
    experts = {}

    def layer_weights(wspec, layer, rd):
        return bfp.quantize_state(orig_weights(wspec, layer, rd), spec)

    def build_layer(tc, layer, sd):
        mod = orig_build(tc, layer, sd)
        if rows:
            F.swap_hook(mod.mlp.gate, set(rows), experts)
        return mod

    H.layer_weights, H.build_layer = layer_weights, build_layer
    try:
        base = {k: v for k, v in req.items() if k not in ("quant", "rows")}
        assert "x" in base, "quantized references run on a given input stream"
        res = H.run(base)
    finally:
        H.layer_weights, H.build_layer = orig_weights, orig_build
    res.pop("x", None)
    if rows:
        y = res["y"]
        assert y.shape[0] == 1, "one sequence per reference"
        return {
            "rows": torch.tensor(rows, dtype=torch.int64),
            "y_swap": y[0, rows].clone(),
            "experts": torch.tensor([experts[r] for r in rows], dtype=torch.int64).reshape(-1, 2),
            "seconds": time.time() - t0,
        }
    res["quant"] = spec
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    req = torch.load(args.request, weights_only=False)
    res = run(req)
    tmp = args.out + ".tmp"
    torch.save(res, tmp)
    os.replace(tmp, args.out)
    print(f"hf_quant_ref layer {req['layer']} done in {res['seconds']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
