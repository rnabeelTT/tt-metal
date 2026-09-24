# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Model args for Qwen/Qwen3.5-122B-A10B.

Subclasses the Qwen3.5/3.6 hybrid ``Qwen36ModelArgs`` (Gated DeltaNet + gated attention, partial
RoPE, layer types) and adds the MoE fields. The dense-MLP dimension the base class expects
(``intermediate_size`` / ``hidden_dim``) is set to the SHARED expert width (1024): the shared expert
is exactly a dense SwiGLU of that size, so every dense-MLP program config the base derives applies to
it unchanged. Routed-expert shapes come from ``moe_intermediate_size`` and ``num_experts``.
"""
from __future__ import annotations

import os

from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs

HF_MODEL_ID = "Qwen/Qwen3.5-122B-A10B"


class Qwen35MoeModelArgs(Qwen36ModelArgs):
    """Qwen3.5-122B-A10B ModelArgs (single device for functional bringup, (1,4) TP mesh for serving)."""

    def __init__(self, mesh_device=None, max_batch_size=1, max_seq_len=2048, **kwargs):
        os.environ.setdefault("HF_MODEL", HF_MODEL_ID)
        super().__init__(mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len, **kwargs)
        text_config = self.hf_config.get_text_config()
        self.num_experts = int(text_config.num_experts)
        self.num_experts_per_tok = int(text_config.num_experts_per_tok)
        self.moe_intermediate_size = int(text_config.moe_intermediate_size)
        self.shared_expert_intermediate_size = int(text_config.shared_expert_intermediate_size)
        self.mlp_only_layers = list(getattr(text_config, "mlp_only_layers", []) or [])
        assert self.hidden_dim == self.shared_expert_intermediate_size, (
            f"dense hidden_dim {self.hidden_dim} must equal shared_expert_intermediate_size "
            f"{self.shared_expert_intermediate_size}"
        )
        # Advertised context (HF max_position_embeddings) — the capability contract target.
        self.hf_max_context = int(text_config.max_position_embeddings)

    def _set_params_from_dict(self, config):
        config = dict(config)
        # tt_transformers ModelArgs needs a dense-MLP width and the tt_transformers MoE field name.
        config.setdefault("intermediate_size", config["shared_expert_intermediate_size"])
        config.setdefault("num_local_experts", config["num_experts"])
        super()._set_params_from_dict(config)

    def is_moe_layer(self, layer_idx: int) -> bool:
        return layer_idx not in self.mlp_only_layers

    # (1,4) TP: the framework rule keeps dim <= 4096 on a non-distributed prefill norm (gather before the norm). Setting
    # ``prefill_norm_distributed`` runs the distributed prefill norm instead (fractured output, 768/die = 24 tiles), which
    # re-enables the fused all-gather+matmul in-projections (tp_common.prefill_norm_agmm_enabled). Measured per stage 4.
    prefill_norm_distributed: bool = False

    def is_distributed_norm(self, mode):
        from models.tt_transformers.tt.common import Mode

        if mode == Mode.PREFILL and getattr(self, "prefill_norm_distributed", False) and self.is_multichip:
            return True
        return super().is_distributed_norm(mode)

    def load_state_dict(self):
        """Streaming state dict (the 250 GB checkpoint never fits in host RAM)."""
        from models.autoports.qwen_qwen3_5_122b_a10b.tt.weights import StreamingStateDict

        return StreamingStateDict(self.CKPT_DIR, linear_q_dim=self.linear_q_dim, linear_k_dim=self.linear_k_dim)
