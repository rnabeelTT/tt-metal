# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic (HF venv, never imports ttnn): the HF decoder layer in fp32 on a cached bf16 reference's input stream,
with the sub-block taps of hf_layer_ref.py ("taps": True), plus the fp32 FFN and its routed / shared parts on HF's
bf16 FFN input (isolates the FFN from upstream rounding). The bf16 reference is the HF layer as the checkpoint
loads it; this run is the same layer with every weight upcast to fp32 (no new information, only no rounding).

usage: python -I hf_fp32_taps.py --base <cached reference .pt with "x" and "taps"> --layer L --segments '[n]' --out <.pt>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--segments", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    base = torch.load(a.base, weights_only=False, map_location="cpu")
    rd = H.ckpt.CheckpointReader()
    tc = H.text_config(rd.path, rd.num_experts)
    mod = H.build_layer(tc, a.layer, H.layer_weights("real", a.layer, rd)).float()
    taps = {}

    def tap(name, which):
        def hook(m, args, out):
            if name in taps:
                return
            o = out[0] if isinstance(out, tuple) and which == "out0" else out
            if which == "in0":
                o = args[0]
            if which == "hc":
                o = out[2]
            taps[name] = o.detach().clone()

        return hook

    mod.attn_hc.register_forward_hook(tap("attn_collapsed", "hc"))
    mod.input_layernorm.register_forward_hook(tap("attn_in", "out"))
    mod.self_attn.register_forward_hook(tap("attn_out", "out0"))
    mod.ffn_hc.register_forward_hook(tap("X1", "in0"))
    mod.ffn_hc.register_forward_hook(tap("ffn_collapsed", "hc"))
    mod.post_attention_layernorm.register_forward_hook(tap("ffn_in", "out"))
    mod.mlp.register_forward_hook(tap("ffn_out", "out"))
    x = base["x"].float()
    segs = json.loads(a.segments)
    cache = DynamicCache(config=tc)
    ys, s0 = [], 0
    for seg in segs:
        y, _ = mod(
            x[:, s0 : s0 + seg],
            attention_mask=torch.ones(1, seg, dtype=torch.bool),
            past_key_values=cache,
            use_cache=True,
            position_embeddings=None,
        )
        ys.append(y)
        s0 += seg
    res = {"taps": taps, "y": torch.cat(ys, 1)}
    # fp32 FFN (and its routed / shared parts) on HF's bf16 FFN input
    fin = base["taps"]["ffn_in"].float()
    n = fin.shape[1]
    res["ffn_out_on_hf_in"] = mod.mlp(fin)
    if hasattr(mod.mlp, "shared_experts"):
        res["shared_on_hf_in"] = mod.mlp.shared_experts(fin)
        res["routed_on_hf_in"] = res["ffn_out_on_hf_in"] - res["shared_on_hf_in"]
    torch.save(res, a.out)
    print(f"hf_fp32_taps layer {a.layer}: {n} rows, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
