# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Ornith decoder layer: the Qwen3.5 hybrid layer with an MoE feed-forward.

Everything else in the layer is unchanged from the qwen36 port -- Gated DeltaNet,
gated full attention, partial/m-RoPE, zero-centred RMSNorm, the residual pattern, the
tensor-parallel norm fusions. Only the feed-forward differs, so this subclass overrides
exactly that hook and inherits the rest.
"""
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.moe import OrnithMoE
from models.demos.blackhole.qwen36.tt.layer import Qwen36DecoderLayer
from models.demos.blackhole.qwen36.utils.substate import substate


class OrnithDecoderLayer(Qwen36DecoderLayer):
    """Qwen3.5 hybrid layer whose MLP is a 256-expert top-8 MoE block."""

    def _build_feed_forward(self, mesh_device, args, state_dict, layer_num, tensor_cache_path, tt_ccl):
        moe_state = substate(state_dict, f"layers.{layer_num}.mlp")
        cache = (tensor_cache_path / f"layers.{layer_num}") if tensor_cache_path else None
        # tt_ccl carries the collective context the tensor-parallel reduce needs.
        return OrnithMoE(mesh_device, args, moe_state, layer_id=layer_num, tensor_cache_path=cache, tt_ccl=tt_ccl)
