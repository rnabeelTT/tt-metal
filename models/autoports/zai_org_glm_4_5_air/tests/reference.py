# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HF reference helpers for the GLM-4.5-Air decoder-layer tests (host side only).

The HF layer (``Glm4MoeDecoderLayer``) holds the routed experts fused (``mlp.experts.gate_up_proj`` [E, 2I, H],
``mlp.experts.down_proj`` [E, H, I]); the checkpoint stores them per expert, so :func:`real_layer_state` fuses them.
Every layer is full attention; the layer KIND used for synthetic weights is ``dense`` (layer < first_k_dense_replace)
or ``moe``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeDecoderLayer, Glm4MoeRotaryEmbedding

from models.autoports.zai_org_glm_4_5_air.tt.weights import HF_MODEL_ID, hf_layer_state, resolve_checkpoint_dir

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
    "layer_kind",
    "new_kv_cache",
]


def new_kv_cache():
    return DynamicCache()


def text_config():
    ckpt = resolve_checkpoint_dir(os.environ.get("HF_MODEL", HF_MODEL_ID))
    cfg = AutoConfig.from_pretrained(str(ckpt), local_files_only=True)
    text = cfg.get_text_config()
    text._attn_implementation = "eager"
    if getattr(text, "layer_types", None) is None:
        text.layer_types = ["full_attention"] * int(text.num_hidden_layers)
    return cfg, text


class LinearCache:  # kept for import compatibility of the shared test modules (no GDN layer in GLM)
    def __init__(self, layer_idx: int):
        raise RuntimeError("GLM-4.5-Air has no linear-attention layers")


def layer_kind(text, layer_idx: int) -> str:
    return "dense" if layer_idx < int(getattr(text, "first_k_dense_replace", 0)) else "moe"


def _fuse_experts(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out, per = {}, {}
    for k, t in state.items():
        p = k.split(".")
        if p[:2] == ["mlp", "experts"] and p[2].isdigit():
            per[(int(p[2]), p[3])] = t
        else:
            out[k] = t
    if per:
        n = 1 + max(e for e, _ in per)
        out["mlp.experts.gate_up_proj"] = torch.stack(
            [torch.cat([per[(e, "gate_proj")], per[(e, "up_proj")]], 0) for e in range(n)]
        )
        out["mlp.experts.down_proj"] = torch.stack([per[(e, "down_proj")] for e in range(n)])
    return out


def _hf_layer_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Layer-local HF keys from either scheme: strips a ``layers.N.`` / ``mtp.layers.0.`` prefix, drops the MTP head's
    own ``mtp.*`` tensors (fc / norms / head), and renames the internal ``mlp.shared_expert.`` back to HF's
    ``mlp.shared_experts.``."""
    out = {}
    for k, t in state.items():
        if k.startswith("mtp.layers.0."):
            k = k[len("mtp.layers.0.") :]
        elif k.startswith("mtp."):
            continue
        elif k.startswith("layers.") and k.split(".")[1].isdigit():
            k = k.split(".", 2)[2]
        if k.startswith("mlp.shared_expert."):
            k = "mlp.shared_experts." + k[len("mlp.shared_expert.") :]
        out[k] = t
    return out


def torch_layer(text, layer_idx: int, state: dict[str, torch.Tensor]) -> Glm4MoeDecoderLayer:
    """HF layer from HF-scheme layer-local keys or internal-scheme keys (``layers.N.*``, ``mtp_state()``)."""
    state = _hf_layer_keys(state)
    layer = Glm4MoeDecoderLayer(text, layer_idx=layer_idx)
    missing, unexpected = layer.load_state_dict(_fuse_experts(state), strict=False)
    assert not missing, missing
    assert not unexpected, unexpected
    layer = layer.to(torch.bfloat16).eval()
    if hasattr(layer.mlp, "gate"):  # HF keeps the selection bias in fp32 (_keep_in_fp32_modules_strict)
        layer.mlp.gate.e_score_correction_bias.data = state["mlp.gate.e_score_correction_bias"].to(torch.float32)
    return layer


def rotary(text, hidden: torch.Tensor, position_ids: torch.Tensor):
    return Glm4MoeRotaryEmbedding(text)(hidden, position_ids)


def causal_mask(batch: int, seq_len: int):
    m = torch.full((batch, 1, seq_len, seq_len), torch.finfo(torch.float32).min, dtype=torch.float32)
    return torch.triu(m, diagonal=1).to(torch.bfloat16)


def real_layer_state(layer_idx: int) -> dict[str, torch.Tensor]:
    """HF-format layer-local tensors with the routed experts fused (HF module layout)."""
    return _fuse_experts(hf_layer_state(resolve_checkpoint_dir(os.environ.get("HF_MODEL", HF_MODEL_ID)), layer_idx))


def layer_shapes(text, layer_idx: int) -> dict[str, tuple[int, ...]]:
    """HF tensor shapes for one layer (HF module layout; fused routed experts)."""
    H = text.hidden_size
    q = text.num_attention_heads * text.head_dim
    kv = text.num_key_value_heads * text.head_dim
    shapes = {
        "input_layernorm.weight": (H,),
        "post_attention_layernorm.weight": (H,),
        "self_attn.q_proj.weight": (q, H),
        "self_attn.q_proj.bias": (q,),
        "self_attn.k_proj.weight": (kv, H),
        "self_attn.k_proj.bias": (kv,),
        "self_attn.v_proj.weight": (kv, H),
        "self_attn.v_proj.bias": (kv,),
        "self_attn.o_proj.weight": (H, q),
    }
    if layer_kind(text, layer_idx) == "dense":
        D = text.intermediate_size
        shapes.update({"mlp.gate_proj.weight": (D, H), "mlp.up_proj.weight": (D, H), "mlp.down_proj.weight": (H, D)})
    else:
        E, N = text.n_routed_experts, text.moe_intermediate_size
        S = N * text.n_shared_experts
        shapes.update(
            {
                "mlp.gate.weight": (E, H),
                "mlp.gate.e_score_correction_bias": (E,),
                "mlp.experts.gate_up_proj": (E, 2 * N, H),
                "mlp.experts.down_proj": (E, H, N),
                "mlp.shared_experts.gate_proj.weight": (S, H),
                "mlp.shared_experts.up_proj.weight": (S, H),
                "mlp.shared_experts.down_proj.weight": (H, S),
            }
        )
    return shapes


def synthetic_layer_state(text, layer_idx: int, seed: int = 0) -> dict[str, torch.Tensor]:
    """Deterministic synthetic weights from recorded real-weight statistics (tests/weight_stats.json)."""
    stats = json.loads(STATS_PATH.read_text())[layer_kind(text, layer_idx)]
    out = {}
    for i, (name, shape) in enumerate(sorted(layer_shapes(text, layer_idx).items())):
        st = stats[name]
        gen = torch.Generator().manual_seed(seed * 1000 + i)
        t = torch.randn(shape, generator=gen, dtype=torch.float32) * st["std"] + st["mean"]
        out[name] = t.to(torch.float32 if name.endswith("e_score_correction_bias") else torch.bfloat16)
    return out


def collect_weight_stats(layers=(0, 1)) -> dict:
    """Record name/shape/dtype/mean/std/min/max of the real weights of one layer per kind."""
    _, text = text_config()
    stats = {}
    for layer_idx in layers:
        kind = layer_kind(text, layer_idx)
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
