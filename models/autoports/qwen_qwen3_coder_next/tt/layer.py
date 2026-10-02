# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-122B-A10B decoder layer: the qwen36 hybrid layer with the MoE feed-forward."""
from models.autoports.qwen_qwen3_coder_next.tt.moe import Qwen35MoE
from models.demos.blackhole.qwen36.tt.layer import Qwen36DecoderLayer
from models.demos.blackhole.qwen36.utils.substate import substate


class Qwen35MoeDecoderLayer(Qwen36DecoderLayer):
    """GDN / gated-attention token mixer from qwen36; feed-forward = router + routed + shared experts."""

    def _build_feed_forward(self, mesh_device, args, state_dict, layer_num, tensor_cache_path, tt_ccl):
        if not args.is_moe_layer(layer_num):
            return super()._build_feed_forward(mesh_device, args, state_dict, layer_num, tensor_cache_path, tt_ccl)
        mlp_state = substate(state_dict, f"layers.{layer_num}.mlp")
        mlp_cache = (tensor_cache_path / f"layers.{layer_num}") if tensor_cache_path else None
        return Qwen35MoE(mesh_device, args, mlp_state, mlp_cache, tt_ccl=tt_ccl, layer_num=layer_num)
