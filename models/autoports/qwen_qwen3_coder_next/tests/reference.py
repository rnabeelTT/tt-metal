# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HF reference helpers for the Qwen3-Coder-Next decoder-layer tests (host side only)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextDecoderLayer, Qwen3NextRotaryEmbedding

from models.autoports.qwen_qwen3_coder_next.tt.weights import HF_MODEL_ID, hf_layer_state, resolve_checkpoint_dir

STATS_PATH = Path(__file__).with_name("weight_stats.json")

# Re-exported for the test modules (keeps autoflake from dropping the import).
__all__ = [
    "DynamicCache",
    "LinearCache",
    "text_config",
    "torch_layer",
    "rotary",
    "causal_mask",
    "real_layer_state",
    "synthetic_layer_state",
    "collect_weight_stats",
    "layer_shapes",
    "new_kv_cache",
]


def new_kv_cache():
    return DynamicCache()


def text_config():
    ckpt = resolve_checkpoint_dir(os.environ.get("HF_MODEL", HF_MODEL_ID))
    cfg = AutoConfig.from_pretrained(str(ckpt), trust_remote_code=True, local_files_only=True)
    text = cfg.get_text_config()
    text._attn_implementation = "eager"
    return cfg, text


class _LinearLayerState:
    def __init__(self):
        self.conv_states = None
        self.recurrent_states = None


class LinearCache:
    """Minimal cache object for Qwen3NextGatedDeltaNet (conv + recurrent state per layer)."""

    def __init__(self, layer_idx: int):
        self.layers = [_LinearLayerState() for _ in range(layer_idx + 1)]

    def has_previous_state(self, layer_idx: int) -> bool:
        s = self.layers[layer_idx]
        return s.conv_states is not None and s.recurrent_states is not None

    def update_conv_state(self, new_conv_state, layer_idx: int):
        self.layers[layer_idx].conv_states = new_conv_state.detach().clone()

    def update_recurrent_state(self, recurrent_state, layer_idx: int):
        self.layers[layer_idx].recurrent_states = recurrent_state.detach().clone()


def torch_layer(text, layer_idx: int, state: dict[str, torch.Tensor]) -> Qwen3NextDecoderLayer:
    layer = Qwen3NextDecoderLayer(text, layer_idx=layer_idx)
    missing, unexpected = layer.load_state_dict(state, strict=False)
    assert not missing, missing
    assert not unexpected, unexpected
    return layer.to(torch.bfloat16).eval()


def rotary(text, hidden: torch.Tensor, position_ids: torch.Tensor):
    return Qwen3NextRotaryEmbedding(text)(hidden, position_ids)


def causal_mask(batch: int, seq_len: int):
    m = torch.full((batch, 1, seq_len, seq_len), torch.finfo(torch.float32).min, dtype=torch.float32)
    return torch.triu(m, diagonal=1).to(torch.bfloat16)


def real_layer_state(layer_idx: int) -> dict[str, torch.Tensor]:
    return hf_layer_state(resolve_checkpoint_dir(os.environ.get("HF_MODEL", HF_MODEL_ID)), layer_idx)


def layer_shapes(text, layer_idx: int) -> dict[str, tuple[int, ...]]:
    """HF tensor shapes for one layer (from config; the real checkpoint is the canonical contract)."""
    H, E, N, S = text.hidden_size, text.num_experts, text.moe_intermediate_size, text.shared_expert_intermediate_size
    shapes = {
        "input_layernorm.weight": (H,),
        "post_attention_layernorm.weight": (H,),
        "mlp.gate.weight": (E, H),
        "mlp.experts.gate_up_proj": (E, 2 * N, H),
        "mlp.experts.down_proj": (E, H, N),
        "mlp.shared_expert.gate_proj.weight": (S, H),
        "mlp.shared_expert.up_proj.weight": (S, H),
        "mlp.shared_expert.down_proj.weight": (H, S),
        "mlp.shared_expert_gate.weight": (1, H),
    }
    if text.layer_types[layer_idx] == "full_attention":
        q = text.num_attention_heads * text.head_dim
        kv = text.num_key_value_heads * text.head_dim
        shapes.update(
            {
                "self_attn.q_proj.weight": (2 * q, H),
                "self_attn.k_proj.weight": (kv, H),
                "self_attn.v_proj.weight": (kv, H),
                "self_attn.o_proj.weight": (H, q),
                "self_attn.q_norm.weight": (text.head_dim,),
                "self_attn.k_norm.weight": (text.head_dim,),
            }
        )
    else:
        kd = text.linear_num_key_heads * text.linear_key_head_dim
        vd = text.linear_num_value_heads * text.linear_value_head_dim
        shapes.update(
            {
                "linear_attn.A_log": (text.linear_num_value_heads,),
                "linear_attn.dt_bias": (text.linear_num_value_heads,),
                "linear_attn.conv1d.weight": (2 * kd + vd, 1, text.linear_conv_kernel_dim),
                "linear_attn.in_proj_qkvz.weight": (2 * kd + 2 * vd, H),
                "linear_attn.in_proj_ba.weight": (2 * text.linear_num_value_heads, H),
                "linear_attn.norm.weight": (text.linear_value_head_dim,),
                "linear_attn.out_proj.weight": (H, vd),
            }
        )
    return shapes


def synthetic_layer_state(text, layer_idx: int, seed: int = 0) -> dict[str, torch.Tensor]:
    """Deterministic synthetic weights from recorded real-weight statistics (tests/weight_stats.json)."""
    kind = text.layer_types[layer_idx]
    stats = json.loads(STATS_PATH.read_text())[kind]
    out = {}
    for i, (name, shape) in enumerate(sorted(layer_shapes(text, layer_idx).items())):
        st = stats[name]
        gen = torch.Generator().manual_seed(seed * 1000 + i)
        t = torch.randn(shape, generator=gen, dtype=torch.float32) * st["std"] + st["mean"]
        if name == "linear_attn.A_log":  # keep decay rates in the real range
            t = t.clamp(min=st["min"], max=st["max"])
        out[name] = t.to(torch.bfloat16)
    return out


def collect_weight_stats(layers=(0, 3)) -> dict:
    """Record name/shape/dtype/mean/std/min/max of the real weights of one layer per kind."""
    _, text = text_config()
    stats = {}
    for layer_idx in layers:
        kind = text.layer_types[layer_idx]
        state = real_layer_state(layer_idx)
        stats[kind] = {
            name: {
                "shape": list(t.shape),
                "dtype": str(t.dtype).replace("torch.", ""),
                "mean": float(t.float().mean()),
                "std": float(t.float().std()),
                "min": float(t.float().min()),
                "max": float(t.float().max()),
                "source_layer": layer_idx,
            }
            for name, t in state.items()
        }
    return stats
