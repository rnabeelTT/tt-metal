# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Inkling-Small model on the qwen35 autoport framework (exports ``Qwen35MoeModel`` / ``create_model``, the names the
hybrid overlay and the serving glue import).

The M2.7 port's model class with the Inkling decoder layer. Model-level differences are folded into the weights by
tt/weights.StreamingStateDict (normed embedding table, final norm / 16, dense global_scale); there is no RoPE (the
framework's RoPE tables are built at a dummy width and never read by the layers). The 4 short-conv states per layer
are per-sequence: reset_sequence_state() before a new sequence (the serving glue calls it on prefill).
"""

from __future__ import annotations

import json
import os

from loguru import logger

from models.autoports.minimaxai_minimax_m2_7.tt import precision
from models.autoports.minimaxai_minimax_m2_7.tt.model import Qwen35MoeModel as _M27Model
from models.autoports.minimaxai_minimax_m2_7.tt.model import _CacheRoot
from models.autoports.minimaxai_minimax_m2_7.tt.multichip_decoder import MULTICHIP_SELECTED, MultichipDecoder
from models.autoports.thinkingmachines_inkling_small.tt.layer import InklingDecoderLayer
from models.autoports.thinkingmachines_inkling_small.tt.model_config import Qwen35MoeModelArgs, check_mesh_shape

ROPE_DUMMY_DIM = 64  # the framework builds RoPE tables at args.rope_head_dim; Inkling's layers never read them


class Qwen35MoeModel(_M27Model):
    DECODER_LAYER_CLS = InklingDecoderLayer

    def __init__(self, mesh_device, args, state_dict, tensor_cache_path=None):
        if not args.rope_head_dim:
            args.rope_head_dim = ROPE_DUMMY_DIM
            args.partial_rotary_factor = ROPE_DUMMY_DIM / args.head_dim
        super().__init__(mesh_device, args, state_dict, tensor_cache_path)

    def reset_sequence_state(self):
        """Zero every layer's short-conv states (a new sequence)."""
        for layer in self.layers:
            layer.reset()


def create_model(mesh_device, *, max_batch_size=1, max_seq_len=None, n_layers=None, layer_indices=None, mc=None):
    """(args, model) for the mesh; ``layer_indices`` selects a reduced model (probes, layer tests)."""
    mc = mc or MULTICHIP_SELECTED
    policy, policy_path = precision.load_policy(num_devices=mesh_device.get_num_devices())
    mc = precision.apply_policy(mc, policy)
    logger.info(f"precision policy {policy.get('config_id')} ({policy_path or 'code defaults'}): "
                f"{json.dumps(precision.describe_config(mc), sort_keys=True)}")
    check_mesh_shape(mesh_device)
    args = Qwen35MoeModelArgs(mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len or 32768)
    MultichipDecoder._args_hook(args, mc, mc.to_optimization_config(), mesh_device)
    if layer_indices is not None:
        args.layer_indices = [int(i) for i in layer_indices]
        args.n_layers = len(args.layer_indices)
    elif n_layers is not None:
        args.n_layers = int(n_layers)
        args.attention_type_list = args.attention_type_list[: int(n_layers)]
    use_cache = os.environ.get("QWEN35_TENSOR_CACHE") == "1"
    cache_path = _CacheRoot(args.weight_cache_path(), disabled=not use_cache)
    state_dict = args.load_state_dict()
    model = Qwen35MoeModel(mesh_device, args, state_dict, tensor_cache_path=cache_path)
    model.mc = mc
    model.precision_policy = policy
    return args, model
