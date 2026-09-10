# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint key remapping: Qwen3.5-MoE naming -> what the qwen36 port expects.

The qwen36 modules were written against the dense Qwen3.5/3.6 checkpoints. The MoE
checkpoint stores the same tensors under two different names in its Gated DeltaNet
block, so the weights are remapped once at load rather than by forking the GDN loader:

  in_proj_qkv.weight  ->  qkv_proj.weight      (a pure rename)
  conv1d.weight       ->  q_conv / k_conv / v_conv .weight

The conv is the interesting one. This checkpoint keeps ONE depthwise conv over the
concatenated q|k|v stream; the port wants it as three, split at the head boundaries
q = num_key_heads * key_head_dim, k likewise, v = num_value_heads * value_head_dim.
Splitting anywhere else silently mixes one projection's taps into another -- the model
still runs and still emits fluent text, so the split points are asserted against the
config rather than hard-coded.
"""
from __future__ import annotations

import re

import torch

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def gdn_split_sizes(text_config) -> tuple[int, int, int]:
    """(q, k, v) channel counts of the fused DeltaNet conv/projection."""
    q = text_config["linear_num_key_heads"] * text_config["linear_key_head_dim"]
    k = q
    v = text_config["linear_num_value_heads"] * text_config["linear_value_head_dim"]
    return q, k, v


def remap_ornith_state_dict(state_dict: dict[str, torch.Tensor], text_config) -> dict[str, torch.Tensor]:
    """Return a new state dict using the names the qwen36 modules load.

    Keys may be rooted anywhere (``layers.N.…`` or ``model.language_model.layers.N.…``);
    only the trailing ``linear_attn.*`` portion is rewritten, so callers do not have to
    agree on a prefix.
    """
    q, k, v = gdn_split_sizes(text_config)
    out: dict[str, torch.Tensor] = {}
    for key, t in state_dict.items():
        if key.endswith("linear_attn.in_proj_qkv.weight"):
            out[key.replace("in_proj_qkv.weight", "qkv_proj.weight")] = t
        elif key.endswith("linear_attn.conv1d.weight"):
            base = key[: -len("conv1d.weight")]
            if t.shape[0] != q + k + v:
                raise ValueError(
                    f"{key}: fused conv has {t.shape[0]} channels but the config implies "
                    f"q+k+v = {q}+{k}+{v} = {q + k + v}. Splitting on the wrong boundary "
                    f"mixes projections together and does not fail loudly at runtime."
                )
            qw, kw, vw = torch.split(t, [q, k, v], dim=0)
            out[f"{base}q_conv.weight"] = qw.contiguous()
            out[f"{base}k_conv.weight"] = kw.contiguous()
            out[f"{base}v_conv.weight"] = vw.contiguous()
        else:
            out[key] = t
    return out


def load_ornith_state_dict(ckpt_dir, text_config) -> dict[str, torch.Tensor]:
    """Load the full checkpoint and put it in the form the port's modules expect.

    Two remaps compose here:
      1. `remap_qwen36_state_dict` -- strips the `model.language_model.` prefix, renames
         the top-level tensors (embed_tokens -> tok_embeddings, lm_head -> output), and
         drops the vision tower and the MTP layer. The last two are exactly the
         text-first scope, so no extra filtering is needed.
      2. `remap_ornith_state_dict` -- the MoE checkpoint's DeltaNet naming.

    Shards are found through model.safetensors.index.json rather than a filename glob:
    the shard naming differs between checkpoints, and a glob that silently matches
    nothing yields an empty state dict and a model of random weights.
    """
    import json
    import pathlib

    from safetensors import safe_open

    from models.demos.blackhole.qwen36.tt.weight_mapping import remap_qwen36_state_dict

    ckpt = pathlib.Path(ckpt_dir)
    index = json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for key, shard in index.items():
        by_shard.setdefault(shard, []).append(key)

    raw: dict[str, torch.Tensor] = {}
    for shard, keys in sorted(by_shard.items()):
        with safe_open(ckpt / shard, framework="pt", device="cpu") as f:
            for key in keys:
                raw[key] = f.get_tensor(key)
    if not raw:
        raise RuntimeError(f"no tensors loaded from {ckpt}")
    return remap_ornith_state_dict(remap_qwen36_state_dict(raw), text_config)
