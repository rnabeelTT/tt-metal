#!/usr/bin/env python3
"""Generate and cache HF reference outputs for the full model.

Running the 35B reference costs ~70 GB of host RAM and minutes per prompt, so it is done
ONCE here and cached. Every model-level test then grades against the same fixed artifact
instead of re-running the reference, which keeps the device tests fast and makes their
results comparable across runs.

Saves, per prompt: the input ids, the final-position logits, and the top-8 next tokens.
Top-k tokens are the check that actually discriminates -- a model with a broken router
still produces smooth logits, but it picks different tokens.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import torch

# Every prompt is padded out to at least MIN_TOKENS of real text.
#
# The GDN chunk-prefill kernel operates on 128-token chunks, and a shorter sequence does
# not simply run slower -- the model forward dies with "statically allocated circular
# buffers clash with L1 buffers" inside the GDN op, an error that names neither the
# sequence length nor the kernel that cares about it. Padding with a repeated preamble
# (rather than pad tokens) keeps the input a real sequence, so the next-token prediction
# at the end is still a meaningful thing to grade.
# EXACTLY this many tokens, not "at least".
#   * below 128 the GDN chunk-prefill kernel's circular buffers do not fit;
#   * the tensor-parallel GDN path additionally requires T to be a multiple of its
#     chunk size ("OPT-A flat v requires T (138) to be a multiple of chunk_size (32)"),
#     which the single-device path tolerates and multi-chip does not.
# 128 satisfies both. The prompt is padded with a preamble and then LEFT-truncated, so
# the real question stays at the tail where the next-token prediction is made.
TARGET_TOKENS = 128
_PREAMBLE = "The following is a transcript of a careful, factual conversation. " "Each answer is short and correct. "

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "Q: A train leaves at 3pm travelling 60 km/h. How far in 2.5 hours?\nA:",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--layers",
        type=int,
        default=0,
        help="Truncate to this many layers (0 = all). A short stack keeps the "
        "reference cheap while still exercising every module type.",
    )
    a = ap.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.weights)

    cfg = AutoConfig.from_pretrained(a.weights)
    if a.layers:
        cfg.text_config.num_hidden_layers = a.layers
    print(f"loading reference ({cfg.get_text_config().num_hidden_layers} layers, bf16, CPU)…", flush=True)
    model = AutoModelForCausalLM.from_pretrained(a.weights, config=cfg, dtype=torch.bfloat16, device_map="cpu").eval()

    records = []
    for i, prompt in enumerate(PROMPTS):
        ids = tok(prompt, return_tensors="pt").input_ids
        while ids.shape[-1] < TARGET_TOKENS:
            prompt = _PREAMBLE + prompt
            ids = tok(prompt, return_tensors="pt").input_ids
        ids = ids[:, -TARGET_TOKENS:]
        with torch.no_grad():
            logits = model(ids).logits[0, -1].float()
        top = torch.topk(logits, 8)
        records.append(
            {
                "prompt": tok.decode(ids[0]),
                "input_ids": ids[0].tolist(),
                "top_tokens": top.indices.tolist(),
                "top_logits": [round(v, 4) for v in top.values.tolist()],
                "top_text": [tok.decode([t]) for t in top.indices.tolist()],
            }
        )
        torch.save(logits, out / f"logits_{i}.pt")
        print(f"  [{i}] {prompt!r} -> {records[-1]['top_text'][:3]}", flush=True)

    meta = {"layers": cfg.get_text_config().num_hidden_layers, "prompts": records}
    (out / "reference.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}/reference.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
