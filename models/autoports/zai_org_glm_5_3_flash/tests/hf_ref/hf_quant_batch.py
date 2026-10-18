# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Batch runner of quantization-aware HF references (hf_quant_ref.py; runs in the HF venv, never imports ttnn): requests
of the same (layer, weights, quant) share one quantized layer build (the 192-expert load and rounding dominate a
single request). tools/compute_quant_refs.py writes the '<request path> <out path>' list (out = the client cache path
of each request, whose key covers hf_quant_ref.py). A batch result equals hf_quant_ref.run() of the request
(tools/compute_quant_refs.py --check compares one).

usage: .venv_hf/bin/python -I tests/hf_ref/hf_quant_batch.py --batch <list> [--threads N]"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("glm53_hf_quant_ref", HERE / "hf_quant_ref.py")
Q = importlib.util.module_from_spec(_spec)
sys.modules["glm53_hf_quant_ref"] = Q
_spec.loader.exec_module(Q)
H, F, bfp = Q.H, Q.F, Q.bfp


@torch.no_grad()
def run_batch(pairs):
    """Several requests of the same (layer, weights, quant): the quantized layer is built once. The per-request
    forward is that of hf_layer_ref.run() / run() above (fresh cache per request, the router margin, taps of the first
    segment, the near-tie swap on `rows`), with the hooks removed after each request."""
    import json

    from transformers.cache_utils import DynamicCache

    groups = {}
    for rp, op in pairs:
        req = torch.load(rp, weights_only=False)
        key = (int(req["layer"]), json.dumps(req["weights"], sort_keys=True), json.dumps(req["quant"], sort_keys=True))
        groups.setdefault(key, []).append((req, op))
    for (layer, _, _), items in groups.items():
        t0 = time.time()
        req0 = items[0][0]
        ckpt_dir = Path(req0.get("ckpt") or os.environ["GLM53_CKPT"])
        rd = H.ckpt.CheckpointReader(ckpt_dir)
        tc = H.text_config(ckpt_dir, rd.num_experts)
        torch.manual_seed(0)
        sd = bfp.quantize_state(H.layer_weights(req0["weights"], layer, rd), dict(req0["quant"]))
        mod = H.build_layer(tc, layer, sd)
        del sd
        print(f"layer {layer}: quantized layer built in {time.time() - t0:.0f} s, {len(items)} requests", flush=True)
        for req, op in items:
            t1 = time.time()
            rows = sorted(int(r) for r in req.get("rows") or [])
            handles, margins, taps, experts = [], [], {}, {}
            gate = getattr(mod.mlp, "gate", None)
            if gate is not None and hasattr(gate, "e_score_correction_bias"):
                k = gate.top_k

                def mhook(m, args, out, k=k):
                    logits = out[0].detach().float()
                    B = args[0].shape[0] if args and args[0].dim() == 3 else 1
                    sc = torch.sigmoid(logits) + m.e_score_correction_bias.detach().float()
                    top = sc.topk(k + 1, dim=-1).values
                    margins.append((top[..., k - 1] - top[..., k]).reshape(B, -1))

                handles.append(gate.register_forward_hook(mhook))
            if rows:
                handles.append(F.swap_hook(mod.mlp.gate, set(rows), experts))
            if req.get("taps"):

                def tap(name, which):
                    def hook(m, args, out):
                        if name in taps:
                            return
                        o = out[0] if isinstance(out, tuple) and which == "out0" else out
                        if which == "in0":
                            o = args[0] if args else None
                        taps[name] = o.detach().clone() if torch.is_tensor(o) else o

                    return hook

                handles.append(mod.post_attention_layernorm.register_forward_hook(tap("ffn_in", "out")))
                handles.append(mod.mlp.register_forward_hook(tap("ffn_out", "out")))
            x = req["x"].to(torch.bfloat16)
            B = x.shape[0]
            cache = DynamicCache(config=tc)
            ys, s0 = [], 0
            for seg in req["segments"]:
                y, _ = mod(
                    x[:, s0 : s0 + seg],
                    attention_mask=torch.ones(B, seg, dtype=torch.bool),
                    past_key_values=cache,
                    use_cache=True,
                    position_embeddings=None,
                )
                ys.append(y)
                s0 += seg
            for h in handles:
                h.remove()
            y = torch.cat(ys, dim=1)
            if rows:
                res = {
                    "rows": torch.tensor(rows, dtype=torch.int64),
                    "y_swap": y[0, rows].clone(),
                    "experts": torch.tensor([experts[r] for r in rows], dtype=torch.int64).reshape(-1, 2),
                }
            else:
                res = {"y": y, "quant": dict(req["quant"]), "block_type": tc.layer_types[layer]}
                if margins:
                    res["route_margin"] = torch.cat(margins, dim=1)
                if taps:
                    res["taps"] = taps
            res["seconds"] = time.time() - t1
            tmp = op + ".tmp"
            torch.save(res, tmp)
            os.replace(tmp, op)
            print(f"  {Path(op).name}: {res['seconds']:.0f} s", flush=True)
        del mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, help="file of '<request path> <out path>' lines")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    run_batch([ln.split() for ln in Path(args.batch).read_text().splitlines() if ln.strip()])


if __name__ == "__main__":
    main()
