# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Model args for thinkingmachines/Inkling-Small on the qwen35 autoport framework (shared with the MiniMax-M2.7 port,
whose module / class names the hybrid MoE overlay patches: this module exports ``Qwen35MoeModelArgs``).

Inkling differs from the models this framework was written for in ways the args carry as flags (the code that reads
them is tt/attention.py, tt/optimized_decoder.py, tt/model.py):

* attention: no RoPE (a learned relative-position bias), q/k per-head RMSNorm with a plain weight and scale 1/head_dim,
  35 sliding-window (512) layers + 7 global ones, k/v short convolutions -> tt/attention.InklingAttention;
* two more short convolutions per layer (on the attention and the MLP outputs, before the residual adds);
* layers 0-1 dense (I 16384, x global_scale); layers 2-41: 256 routed experts (I 2048), top-6 by sigmoid + bias, plus
  2 shared experts, all 8 weighted by one normalisation (route_scale 8 x global_scale): the shared experts are carried
  as experts 256 and 257, so the MoE is 8 weighted experts of 258 (``num_experts`` 258, ``num_experts_per_tok`` 8,
  ``router_kind`` "inkling");
* RMSNorm after the embedding; the final hidden state / 16 before the LM head; RMSNorm weights plain.
"""

from __future__ import annotations

import json
import os

from models.autoports.minimaxai_minimax_m2_7.tt.model_config import Qwen35MoeModelArgs as _M27Args
from models.autoports.minimaxai_minimax_m2_7.tt.model_config import (  # noqa: F401  (re-exported)
    check_mesh_shape,
    mesh_shape_from_env,
)
from models.autoports.thinkingmachines_inkling_small.tt.hf_config_inkling import register as _register_hf_config
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs

HF_MODEL_ID = "thinkingmachines/Inkling-Small"
_register_hf_config()


class Qwen35MoeModelArgs(_M27Args):
    """Inkling-Small ModelArgs (class name kept: the shared code and the hybrid overlay import it by this name)."""

    attn_gated = False
    attn_qk_norm = True
    attn_qkv_bias = False
    norm_unit_offset = False
    router_kind = "inkling"
    shared_expert_gated = False
    has_shared_expert = False  # the 2 shared experts ride the routed path as experts 256, 257

    def __init__(self, mesh_device=None, max_batch_size=1, max_seq_len=2048, **kwargs):
        os.environ.setdefault("HF_MODEL", HF_MODEL_ID)
        # skip the M2.7 init (it reads MiniMax-only config keys); the shared Qwen36 init reads the common fields
        Qwen36ModelArgs.__init__(self, mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len, **kwargs)
        raw = json.load(open(os.path.join(self.CKPT_DIR, "config.json")))["text_config"]
        self.inkling = raw  # the raw text config (the modules read sliding_window_size, d_rel, rel_extent, ...)
        self.n_routed_experts = int(raw["n_routed_experts"])
        self.n_shared_experts = int(raw["n_shared_experts"])
        self.num_experts = self.n_routed_experts + self.n_shared_experts
        self.top_k_routed = int(raw["num_experts_per_tok"])
        self.num_experts_per_tok = self.top_k_routed + self.n_shared_experts
        self.moe_intermediate_size = int(raw["intermediate_size"])  # the expert width (see tt/weights.py)
        self.dense_intermediate_size = int(raw["dense_intermediate_size"])
        self.shared_expert_intermediate_size = 0
        self.route_scale = float(raw["route_scale"])
        self.routed_scaling_factor = 1.0
        self.norm_topk_prob = True
        self.first_k_dense_replace = int(raw["dense_mlp_idx"])
        self.mlp_only_layers = list(range(self.first_k_dense_replace))
        self.num_nextn_predict_layers = 0  # the MTP heads (mtp.safetensors) are driven by tt/mtp.py
        self.partial_rotary_factor = 0.0
        self.rope_head_dim = 0
        self.attention_type_list = ["full_attention"] * self.n_layers  # every layer is attention (no GDN)
        self.local_layer_ids = set(raw["local_layer_ids"])
        self.hf_max_context = int(raw.get("model_max_length", 1 << 20))
        self.logits_mup_width_multiplier = float(raw["logits_mup_width_multiplier"])
        self.unpadded_vocab_size = int(raw["unpadded_vocab_size"])

    def is_sliding_layer(self, layer_idx: int) -> bool:
        return layer_idx in self.local_layer_ids

    def is_moe_layer(self, layer_idx: int) -> bool:
        return layer_idx >= self.first_k_dense_replace

    def load_state_dict(self):
        from models.autoports.thinkingmachines_inkling_small.tt.weights import StreamingStateDict

        return StreamingStateDict(self.CKPT_DIR)
