# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Random-weight qwen4_exp checkpoints of any size, with golden logits, from the Transformers reference.

The upstream port (models/demos/blackhole/qwen38_flash_next) pins Qwen3.8-Flash-Next's exact config, index SHA-256
and a 1x4 mesh, so its oracle cannot score a smaller Qwen4. This builds the HF ``qwen4_exp`` model (Transformers
>= 5.16.1, CPU) at a chosen size, writes it in the released checkpoint's tensor naming
(``model.language_model.*``, ``mtp.*``, fused ``mlp.experts.gate_up_proj``), and records golden prefill + decode
logits, so device code for p150 / p300 profiles can be tested before real weights exist. Fields a preset does not
set keep Flash-Next's values. Transformers' CausalLM builds no MTP head, so ``mtp.*`` tensors are not produced.

    python -m models.demos.blackhole.qwen4_prep.mini_reference --preset tiny --out /path/ckpt
    python -m models.demos.blackhole.qwen4_prep.mini_reference --preset tiny --check-names INDEX.json

Presets keep every width a multiple of 32 * TP for TP = 1, 2, 4 and keep Flash-Next's ratios (GR rank = hidden / 8,
PLE dim = hidden, 3 GDN : 1 QSA, QSA budget > prompt so selection is exercised only when the prompt is longer).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

PRESETS = {
    # tiny: seconds on CPU; QSA budget 64 tokens with 4-token blocks, so a 160-token prompt exercises selection.
    "tiny": dict(
        hidden_size=256,
        num_hidden_layers=4,
        hc_count=4,
        hc_lowrank=32,
        linear_num_key_heads=4,
        linear_num_value_heads=12,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=64,
        indexer_budget=64,
        indexer_compress_ratio=4,
        num_experts=16,
        num_experts_per_tok=4,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        ple_layer_ids=[2],
        ple_embed_dim=256,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=4093,
        split_ngram_parts=4,
        vocab_size=4096,
        eos_token_id=4095,
        bos_token_id=4095,
    ),
    # small: Qwen3.5-35B-A3B-like widths (hidden 2048, 256-wide heads) at 4 layers and 64 experts: 2.2B params with the full vocabulary, 16 s and 9 GB RAM on CPU;
    # a TP=2 (p300) / TP=1 (p150) shape test for device code. Real 35B-A3B routing is 256 experts top-8.
    "small": dict(
        hidden_size=2048,
        num_hidden_layers=4,
        hc_count=4,
        hc_lowrank=256,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=256,
        num_experts=64,
        num_experts_per_tok=8,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        ple_layer_ids=[2],
        ple_embed_dim=2048,
        ngram_vocab_size_base=65521,
        split_ngram_parts=8,
        eos_token_id=248044,  # Qwen3.5 / Flash-Next tokenizer
        bos_token_id=248044,
    ),
}


def build_config(preset: str, overrides: dict):
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    # Every field not set by the preset or the overrides keeps Qwen3.8-Flash-Next's value (several qwen4_exp config
    # defaults are None and would not build).
    base = json.loads((Path(__file__).resolve().parent / "flash_next_config.json").read_text())["text_config"]
    for key in ("layer_types", "model_type", "dtype", "transformers_version", "mtp"):
        base.pop(key, None)
    fields = {**base, **PRESETS[preset], **overrides}
    layers = fields["num_hidden_layers"]
    interval = fields.get("full_attention_interval", 4)
    fields.setdefault(
        "layer_types", ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(layers)]
    )
    fields["max_position_embeddings"] = overrides.get("max_position_embeddings", 4096)
    return Qwen4ExpTextConfig(**fields)


def released_name(name: str) -> str:
    """Map Qwen4ExpForCausalLM state-dict names onto the released checkpoint's naming."""
    if name.startswith("model."):
        return "model.language_model." + name[len("model.") :]
    return name


def _pattern(name: str) -> str:
    return re.sub(r"\.\d+\.", ".N.", re.sub(r"shard_\d+", "shard_K", name))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=PRESETS, default="tiny")
    p.add_argument("--set", nargs="*", default=[], help="config overrides, key=json_value")
    p.add_argument("--out", type=Path, help="write config.json, model.safetensors and golden.pt here")
    p.add_argument("--check-names", type=Path, help="a released model.safetensors.index.json to compare names with")
    p.add_argument("--prompt-len", type=int, default=160)
    p.add_argument("--decode-steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM

    overrides = {k: json.loads(v) for k, v in (item.split("=", 1) for item in args.set)}
    config = build_config(args.preset, overrides)
    torch.manual_seed(args.seed)
    model = Qwen4ExpForCausalLM(config).eval()
    with torch.no_grad():  # random init is near-degenerate (identical routers); spread every weight a little
        for name, param in model.named_parameters():
            if param.ndim >= 2 and "ngram_embedding" not in name:
                param.normal_(0.0, 0.5 / param.shape[-1] ** 0.5)
    state = {released_name(k): v.contiguous() for k, v in model.state_dict().items()}
    total = sum(v.numel() for v in state.values())
    print(f"preset {args.preset}: {total / 1e6:.1f}M parameters, {len(state)} tensors")

    if args.check_names:
        released = {_pattern(k) for k in json.loads(args.check_names.read_text())["weight_map"] if ".visual." not in k}
        ours = {_pattern(k) for k in state}
        print(
            f"name patterns: {len(ours & released)} shared, only ours {sorted(ours - released)}, "
            f"only released {sorted(released - ours)}"
        )

    ids = torch.randint(0, config.vocab_size - 1, (1, args.prompt_len), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
        prefill_logits = out.logits[:, -1].float()
        cache, token, steps = out.past_key_values, prefill_logits.argmax(-1, keepdim=True), []
        for _ in range(args.decode_steps):
            step = model(input_ids=token, past_key_values=cache, use_cache=True)
            cache = step.past_key_values
            steps.append((token.item(), step.logits[:, -1].float()))
            token = step.logits[:, -1].argmax(-1, keepdim=True)
        # Decode must agree with a full prefill over the same tokens (cache paths vs. the one-pass path).
        full = torch.cat([ids, torch.tensor([[t for t, _ in steps]])], dim=1)
        ref = model(input_ids=full, use_cache=False).logits[0, -1].float()
    err = (ref - steps[-1][1][0]).abs().max().item() / ref.abs().max().item()
    print(f"greedy tokens {[t for t, _ in steps]}; decode vs full-prefill last logits: max rel err {err:.2e}")

    if args.out:
        from safetensors.torch import save_file

        args.out.mkdir(parents=True, exist_ok=True)
        save_file(state, str(args.out / "model.safetensors"))
        (args.out / "config.json").write_text(json.dumps(config.to_dict(), indent=2, default=str))
        torch.save(
            {"input_ids": ids, "prefill_last_logits": prefill_logits, "decode": steps, "preset": args.preset},
            args.out / "golden.pt",
        )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
