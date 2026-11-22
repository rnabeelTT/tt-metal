# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Calibration capture for HYB_BFP4_ROUND's gptq methods (bfp4_round.py): the exact CPU reference (ref_m27: the FP8
checkpoint dequantized, HF transformers layers in bf16, one layer at a time) over fixed calibration text, saving per
layer the MoE inputs and the routed expert ids as <out>/cal_<L>.pt {"h": [T, H] bf16, "idx": [T, k] uint8}.

The text is plans/bfp4_cal_tokens.npy: 24 x 1024 MiniMax-M2.7 tokens of tt-metal sources (24 tracked .py / .md files,
each file's first 1024 tokens; the recipe plans/bfp4_recipe.json was chosen per layer on a capture of the same kind). Resumes per
layer: the hidden state after the last finished layer is kept as <out>/x.pt until the capture completes.

usage: python bfp4_calib.py <hf snapshot dir> <out dir>      env: REF_THREADS (ref_m27's, default 16)
"""

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TOKENS = os.path.join(HERE, "plans", "bfp4_cal_tokens.npy")


@torch.no_grad()
def main():
    snap, out = sys.argv[1], sys.argv[2]
    sys.path.insert(0, HERE)
    import ref_m27
    from ref_m27 import M

    ref_m27.SNAP = snap
    with open(os.path.join(snap, "model.safetensors.index.json")) as f:
        ref_m27.WMAP = json.load(f)["weight_map"]
    os.makedirs(out, exist_ok=True)
    ids = torch.from_numpy(np.load(TOKENS).astype(np.int64))
    B, T = ids.shape
    cfg = ref_m27.config()
    pos = torch.arange(T).view(1, -1).expand(B, -1)
    state = os.path.join(out, "x.pt")
    if os.path.exists(state):
        s = torch.load(state)
        start, x = s["next"], s["x"]
    else:
        start, x = 0, torch.nn.functional.embedding(ids, ref_m27.get("model.embed_tokens.weight"))
    pe = M.MiniMaxM2RotaryEmbedding(cfg)(x, pos)
    for li in range(start, cfg.num_hidden_layers):
        t0 = time.time()
        layer = ref_m27.load_layer(cfg, li)
        cap = {}
        gate_fwd, mlp_fwd = layer.mlp.gate.forward, layer.mlp.forward

        def rec(h, bias, gate_fwd=gate_fwd):
            logits, w, idx = gate_fwd(h, bias)
            cap["idx"] = idx
            return logits, w, idx

        def mrec(h, mlp_fwd=mlp_fwd):
            cap["h"] = h.reshape(-1, h.shape[-1])
            return mlp_fwd(h)

        layer.mlp.gate.forward, layer.mlp.forward = rec, mrec
        x = layer(x, position_embeddings=pe, attention_mask=None, position_ids=pos)
        del layer
        n = cap["h"].shape[0]
        p = os.path.join(out, f"cal_{li}.pt")
        torch.save({"h": cap["h"].bfloat16(), "idx": cap["idx"].reshape(n, -1).to(torch.uint8)}, p + ".tmp")
        os.replace(p + ".tmp", p)
        torch.save({"next": li + 1, "x": x}, state + ".tmp")
        os.replace(state + ".tmp", state)
        print(f"bfp4 calibration: layer {li} in {time.time() - t0:.0f}s", flush=True)
    os.remove(state)


if __name__ == "__main__":
    main()
