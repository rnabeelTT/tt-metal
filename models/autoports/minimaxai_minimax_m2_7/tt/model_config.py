# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Model args for MiniMaxAI/MiniMax-M2.7 (``minimax_m2``), ported from the GLM-4.5-Air autoport.

Subclasses the Qwen3.5/3.6 ``Qwen36ModelArgs`` (partial RoPE, TP configs) and adds the MoE fields. MiniMax-M2.7
differs from the Qwen3.5 checkpoints this code was written for in ways the args carry as flags:

* every layer is full attention (``attention_type_list`` all ``full_attention``, no Gated DeltaNet);
* attention has no biases and no output gate; q/k RMSNorm runs over the full concatenated q (all heads) and k
  (``attn_qk_norm = "full"``), before the head split and RoPE, so with heads split over dies it needs the sum of
  squares of every die;
* RoPE rotates the first ``rotary_dim`` = 64 of the 128 head dims (the checkpoint gives ``rotary_dim``, not
  ``partial_rotary_factor``, which the base class reads and would default to 1.0);
* RMSNorm weights are plain (``norm_unit_offset`` False);
* the router is sigmoid + ``e_score_correction_bias`` selection with normalized sigmoid weights (``router_kind``),
  no routed scaling, no expert groups;
* no shared expert, no dense layers, and no MTP weights in the checkpoint (the config's ``use_mtp`` is not backed).
"""
from __future__ import annotations

import json
import os

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs

HF_MODEL_ID = "MiniMaxAI/MiniMax-M2.7"

# Prefill SDPA of this model (qwen36 attention/tp.py knobs; an explicit env value still wins): q / k chunks 256 and the
# streaming (bf16-dest) SDPA kernel. 46 full-attention layers make long prompts SDPA-bound: served TTFT at 130K tokens
# 108 -> 84.5 s (q256 + fp32-acc off; q256 + k256 alone 93.4 s), 32K-token teacher forcing vs HF bf16 unchanged (top-1
# 0.84 base / 0.84 q256k256 / 0.875 q256k256acc0, top-5 1.00 all) - 2026-10-09 A/B.
for _k, _v in (
    ("QWEN35_SDPA_PREFILL_Q_CHUNK", "256"),
    ("QWEN35_SDPA_PREFILL_K_CHUNK", "256"),
    ("QWEN35_SDPA_PREFILL_FP32_ACC", "0"),
    # decode MoE: the expert-sorted path from 16 rows (128 experts: many users share experts, the sorted path reads each
    # once). Served A/B on 4 dies, TPOT at 8 / 16 / 32 users (128-token prompts): from 32 rows 76.8 / 127.2 / 106.8 ms,
    # from 16 rows 76.8 / 91.5 / 107.3 ms, from 8 rows 76.4 / - / - ms (2026-10-09)
    ("QWEN35_DECODE_SORTED_MIN_ROWS", "16"),
):
    os.environ.setdefault(_k, _v)

# Supported meshes: (1,4) = P300x2 / P150x4 (TP4), (1,2) = P300 / P150x2 (TP2), (1,1) = P150 (one die, TP modules at degree 1).
SUPPORTED_MESH_SHAPES = ((1, 4), (1, 2), (1, 1))
_TWO_CHIP_MESH_NAMES = (
    "P300",
    "P150X2",
    "P150_X2",
)  # Blackhole 2-chip meshes (a p300 = two p150-class dies on one board)
_ONE_CHIP_MESH_NAMES = ("P150", "P100")  # one Blackhole die


def mesh_shape_from_env() -> tuple[int, int]:
    """(1,1) for MESH_DEVICE=P150 / N150 / P100, (1,2) for P300 / P150x2, (1,4) otherwise (P300x2 / P150x4, the default)."""
    name = os.environ.get("MESH_DEVICE", "P300x2").strip().upper()
    if name in _ONE_CHIP_MESH_NAMES:
        return (1, 1)
    return (1, 2) if name in _TWO_CHIP_MESH_NAMES else (1, 4)


def check_mesh_shape(mesh_device) -> None:
    shape = tuple(mesh_device.shape)
    if shape not in SUPPORTED_MESH_SHAPES:
        raise ValueError(f"mesh {shape} not supported (supported: {SUPPORTED_MESH_SHAPES})")


class Qwen35MoeModelArgs(Qwen36ModelArgs):
    """MiniMax-M2.7 ModelArgs (the class name is kept: the shared Qwen3.5 autoport code imports it by this name)."""

    # architecture flags read by the shared qwen36 layer / attention / MoE code (see the module docstring)
    attn_gated = False
    # "full" (the model); M27_ATTN_QK_NORM=none drops the q / k norms (wrong numerics: perf probes only)
    attn_qk_norm = {"full": "full", "none": False}[os.environ.get("M27_ATTN_QK_NORM", "full")]
    attn_qkv_bias = False
    norm_unit_offset = False
    router_kind = "sigmoid_bias"
    shared_expert_gated = False
    has_shared_expert = False

    def __init__(self, mesh_device=None, max_batch_size=1, max_seq_len=2048, **kwargs):
        os.environ.setdefault("HF_MODEL", HF_MODEL_ID)
        super().__init__(mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len, **kwargs)
        text_config = self.hf_config.get_text_config()
        self.num_experts = int(text_config.num_local_experts)
        self.num_experts_per_tok = int(text_config.num_experts_per_tok)
        self.moe_intermediate_size = int(text_config.intermediate_size)  # MiniMax's per-expert width
        self.shared_expert_intermediate_size = 0
        self.routed_scaling_factor = 1.0
        self.norm_topk_prob = True
        if getattr(text_config, "scoring_func", "sigmoid") != "sigmoid" or not getattr(
            text_config, "use_routing_bias", True
        ):
            raise ValueError(
                f"router {getattr(text_config, 'scoring_func', None)} / use_routing_bias "
                f"{getattr(text_config, 'use_routing_bias', None)}: only sigmoid + bias is implemented"
            )
        self.first_k_dense_replace = 0
        self.mlp_only_layers = []
        self.dense_intermediate_size = 0
        self.num_nextn_predict_layers = 0  # use_mtp is set in the config, but the checkpoint has no MTP weights
        # RoPE on the first rotary_dim dims (the base class read partial_rotary_factor, absent here, as 1.0)
        rotary_dim = int(
            getattr(text_config, "rotary_dim", 0)
            or json.load(open(os.path.join(self.CKPT_DIR, "config.json")))["rotary_dim"]
        )
        self.partial_rotary_factor = rotary_dim / self.head_dim
        self.rope_head_dim = rotary_dim
        # every layer is full attention
        self.attention_type_list = ["full_attention"] * self.n_layers
        # Advertised context (HF max_position_embeddings) — the capability contract target.
        self.hf_max_context = int(text_config.max_position_embeddings)

    def _set_params_from_dict(self, config):
        super()._set_params_from_dict(dict(config))

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
            if topo == ttnn.Topology.Ring and self.num_devices == 2:
                # a 2-device "ring" makes both neighbours the same chip: the fused matmul + ring reduce-scatter of the
                # out-projections deadlocks there (2026-10-02, P300 board, tt-triage: ring_reduce_scatter readers in
                # noc_semaphore_wait_min). Two dies are a line.
                topo = ttnn.Topology.Linear
                logger.info("CCL topology: Linear (2-device mesh)")
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
        from models.autoports.minimaxai_minimax_m2_7.tt.weights import StreamingStateDict

        # the checkpoint's main-layer count (n_layers shrinks for a reduced build; layer num_hidden_layers is the MTP layer)
        return StreamingStateDict(self.CKPT_DIR, num_layers=None)
