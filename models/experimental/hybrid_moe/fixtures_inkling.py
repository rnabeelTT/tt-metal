# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Per-layer fixtures for the Inkling TT port's device tests (the serving image's transformers predates Inkling, so the
device tests compare against tensors produced here with the HF modules and the real BF16 weights).

For each requested layer: runs the residual stream of one text up to that layer (HF decoder layers, ref_inkling's
streaming path), then that layer once more with forward hooks, and saves for the first T tokens:

  x_in      [T, H]  the residual entering the layer
  attn_in   [T, H]  input_layernorm(x_in)           (what the attention sees)
  attn_out  [T, H]  self_attn output (o_proj, before attn_sconv)
  attn_res  [T, H]  attn_sconv(attn_out)            (added to the residual)
  mlp_in    [T, H]  post_attention_layernorm(h)
  mlp_out   [T, H]  MLP / MoE output (before mlp_sconv); MoE layers also: topk_ids [T, 6], topk_w [T, 6], gammas [T, 2]
  x_out     [T, H]  the layer's output
  fp32 except x_in (bf16 as the model carries it).

usage (host venv): python fixtures_inkling.py <texts.json> <text name> <layers, e.g. 0,2,5> <T> <out dir>
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_inkling as R  # noqa: E402
from transformers.models.inkling import modeling_inkling as M  # noqa: E402


@torch.no_grad()
def main():
    texts, name, layers, T, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    layers = sorted(int(x) for x in layers.split(","))
    os.makedirs(out, exist_ok=True)
    ids = torch.tensor([json.load(open(texts))[name][:T]])
    cfg = R.config()
    norm = M.InklingRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    norm.weight.data = R.get("model.llm.embed_norm.weight").to(torch.bfloat16)
    x = norm(torch.nn.functional.embedding(ids, R.get("model.llm.embed.weight")))
    torch.save({"ids": ids[0], "x0": x[0].clone()}, os.path.join(out, "embed.pt"))
    for li in range(max(layers) + 1):
        layer = R.load_layer(cfg, li)
        rec = {}
        if li in layers:
            hooks = []

            def cap(key, which="out"):
                def h(mod, args, kwargs, res):
                    if which == "in":
                        rec[key] = (args[0] if args else kwargs["hidden_states"])[0].float().clone()
                    else:
                        r = res[0] if isinstance(res, tuple) else res
                        rec[key] = r[0].float().clone()

                return h

            def pre(key):
                def h(mod, args, kwargs):
                    rec[key] = (args[0] if args else kwargs["hidden_states"])[0].float().clone()

                return h

            hooks.append(layer.self_attn.register_forward_pre_hook(pre("attn_in"), with_kwargs=True))
            hooks.append(layer.self_attn.register_forward_hook(cap("attn_out"), with_kwargs=True))
            hooks.append(layer.attn_sconv.register_forward_hook(cap("attn_res"), with_kwargs=True))
            hooks.append(layer.mlp.register_forward_pre_hook(pre("mlp_in"), with_kwargs=True))
            hooks.append(layer.mlp.register_forward_hook(cap("mlp_out"), with_kwargs=True))
            if hasattr(layer.mlp, "gate"):
                g = layer.mlp.gate.forward

                def gate_rec(h, g=g):
                    lg, w, idx, sh = g(h)
                    rec.setdefault("topk_ids", []).append(idx)
                    rec.setdefault("topk_w", []).append(w.float())
                    rec.setdefault("gammas", []).append(sh.float())
                    return lg, w, idx, sh

                layer.mlp.gate.forward = gate_rec
            x_in = x
            x = R.run_layer(cfg, layer, x, 0)
            for h in hooks:
                h.remove()
            for k in ("topk_ids", "topk_w", "gammas"):
                if k in rec:
                    rec[k] = torch.cat(rec[k], 0)
            rec["x_in"] = x_in[0].clone()
            rec["x_out"] = x[0].float().clone()
            torch.save(rec, os.path.join(out, f"layer{li}.pt"))
            print(f"layer {li}: saved {sorted(rec)}", flush=True)
        else:
            x = R.run_layer(cfg, layer, x, 0)
            print(f"layer {li}: passed through", flush=True)
        del layer


if __name__ == "__main__":
    main()
