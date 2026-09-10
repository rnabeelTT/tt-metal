# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM entry point for Ornith-1.5-35B-A3B.

The Tenstorrent vLLM plugin finds a model by the HF architecture name in the
`vllm_metadata.json` next to this file, and calls `initialize_vllm_model` on the class
named there. Everything else -- prefill/decode dispatch, paged KV allocation, decode
tracing, the multimodal plumbing -- is inherited from the qwen36 adapter, because the
serving contract is identical: the MoE differs inside a layer, not at the generator
boundary.

Only model construction is overridden, to build an `OrnithModel` with `OrnithModelArgs`
(which carries the per-mesh expert precision and the replicate-attention/shard-experts
parallelism policy) instead of the dense Qwen3.5 model.
"""
from __future__ import annotations

import os
from collections.abc import Mapping

from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeProcessingInfo
from vllm.model_executor.models.qwen3_vl import Qwen3VLDummyInputsBuilder, Qwen3VLMultiModalProcessor
from vllm.multimodal import MULTIMODAL_REGISTRY

from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.model import OrnithModel
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.model_config import OrnithModelArgs
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.weight_mapping import load_ornith_state_dict
from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM


class TT_Qwen3_5MoeProcessingInfo(Qwen3_5MoeProcessingInfo):
    """Processing info for the MoE config, and text-only limits.

    qwen36 registers `TT_Qwen3_5ProcessingInfo`, whose `get_hf_config` asks for the
    DENSE `Qwen3_5Config`; inheriting that registration makes the server die at startup
    with "Invalid type of HuggingFace config. Expected Qwen3_5Config, found
    Qwen3_5MoeConfig". vLLM already ships `Qwen3_5MoeProcessingInfo` for exactly this.
    """

    def get_supported_mm_limits(self) -> Mapping[str, int]:
        # Zero, not one. The checkpoint HAS a vision tower and the inherited adapter
        # knows how to splice its embeddings, but no vision path in this port has been
        # validated against a reference and `initialize_vllm_model` does not build it.
        # Advertising image/video support we have not tested would turn a clear
        # rejection into a runtime failure on a user's request.
        return {"image": 0, "video": 0}


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=TT_Qwen3_5MoeProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class OrnithForConditionalGeneration(Qwen36ForCausalLM):
    """Serving adapter for `Qwen3_5MoeForConditionalGeneration` checkpoints."""

    # Experts are REPLICATED, not sharded (see tt/moe.py: gather makes the expert pair
    # cost the same 79 us at 64 or 256 experts per die, so splitting them bought only
    # an all-reduce per layer, and the whole model is ~18.2 GB against 30 GB per chip).
    # Each die therefore holds a complete model and can serve its own users, which is
    # exactly what vLLM standard data parallelism does -- one engine per rank, its own
    # submesh, no collectives. The plugin blocks standard DP for MoE models by default
    # because a SHARDED MoE would leave each rank without a full expert set; this
    # declares that this model is not that case.
    model_capabilities = {
        **(getattr(Qwen36ForCausalLM, "model_capabilities", None) or {}),
        "supports_standard_dp": True,
        # Every die holds the full KV head set (2 heads), because attention is
        # replicated rather than tensor-parallel. Without this the plugin sizes the
        # paged cache as num_kv_heads // num_devices and the model overruns it.
        "kv_heads_replicated": True,
    }

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        tt_data_parallel=1,
        optimizations=None,
        **kwargs,
    ):
        weights = (
            os.environ.get("MODEL_WEIGHTS_DIR")
            or os.environ.get("ORNITH_WEIGHTS")
            or os.environ.get("HF_MODEL")
            or hf_config._name_or_path
        )
        if weights and not os.path.isdir(os.path.expanduser(weights)):
            from huggingface_hub import snapshot_download

            offline = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("CI") == "true"
            weights = snapshot_download(weights, local_files_only=offline)
        os.environ["ORNITH_WEIGHTS"] = weights
        os.environ.setdefault("HF_MODEL", weights)

        args = OrnithModelArgs(mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len)
        text_config = args.hf_config.get_text_config().to_dict()
        state = load_ornith_state_dict(weights, text_config)

        # A real cache path is required: the embedding builds its cache name as
        # `weight_cache_path / base_name` unconditionally. It also makes the second
        # server start fast -- 40 layers of expert weights reload from cache.
        cache = args.weight_cache_path(args.routed_expert_dtype)
        cache.mkdir(parents=True, exist_ok=True)

        model = OrnithModel(mesh_device, args, state, tensor_cache_path=cache)
        # Text-first: the vision tower is not initialised here. The checkpoint carries
        # one, and `Qwen36ForCausalLM` knows how to splice its embeddings, but no vision
        # path in this port has been validated against a reference yet -- so it is left
        # off rather than served untested.
        return cls([model], [args], mesh_device)
