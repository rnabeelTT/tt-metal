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

import torch
from loguru import logger

import ttnn
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

    def ccl_topology(self):
        """CCL topology for every collective of this model (the layers, the norms, the MTP head and the vision tower
        all read it from here). ``QWEN35_CCL_TOPOLOGY`` = ``auto`` (default) | ``ring`` | ``linear``.

        ``auto``: the stock rule returns Ring for every P300x2 / P150x4 cluster; ``ttnn.get_usable_topology`` demotes
        the Ring request to Linear when the fabric graph the process discovered has no wraparound over the mesh, so the
        decision follows what the fabric can actually route (a 4-chip box cabled as a line, or - the case seen on
        2026-10-01 - a second engine started on this box while another server held the chips: its auto-discovered
        adjacency came out as a LINE ``{1:2, 2:2}`` and the first fused-norm all-gather died with ``TT_FATAL: Could not
        find any forwarding direction from (M0, D0) to (M0, D3)``). Two engines on one device set are never supported
        (tt-model's host-ready check does not detect a live engine); this rule only turns the crash into a Linear run.
        """
        cached = self.__dict__.get("_ccl_topology_cache")
        if cached is not None:
            return cached
        forced = os.environ.get("QWEN35_CCL_TOPOLOGY", "auto").strip().lower()
        if forced in ("ring", "linear"):
            topo = ttnn.Topology.Ring if forced == "ring" else ttnn.Topology.Linear
            logger.info(f"CCL topology forced by QWEN35_CCL_TOPOLOGY={forced}: {topo}")
        else:
            topo = super().ccl_topology()
            if topo == ttnn.Topology.Ring and self.mesh_device is not None and self.num_devices > 1:
                try:
                    probe = ttnn.from_torch(
                        torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.mesh_device,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                    )
                    usable = ttnn.get_usable_topology(probe, ttnn.Topology.Ring)
                    ttnn.deallocate(probe)
                    if usable != ttnn.Topology.Ring:
                        logger.warning(
                            f"CCL topology: the fabric has no ring wraparound over this mesh (usable {usable}); "
                            "collectives run Linear (QWEN35_CCL_TOPOLOGY=ring to force)"
                        )
                        topo = usable
                    else:
                        logger.info("CCL topology: Ring (fabric wraparound confirmed)")
                except Exception as e:  # API drift: keep the stock rule
                    logger.warning(f"CCL topology probe unavailable ({e}); keeping {topo}")
        self._ccl_topology_cache = topo
        return topo

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
