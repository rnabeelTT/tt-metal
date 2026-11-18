# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM adapter for tencent/Hy4-flash (text) on a (1, 4) Blackhole mesh (vllm-tt-plugin, dflash branch).

Single user (``--max-num-seqs 1``). Generation runs in the port's engine (tt/engine.py); the engine owns every cache, so
the KV allocation vLLM asks for is a token budget only and vLLM's prefix caching is off. Prefill returns the first token
(on-device sampling mode) or, when the plugin samples on the host (logprobs, structured outputs, logit bias ...), the
first-token logits; decode returns a host int32 [B, 1] token row (or [B, 1, V] logits for host-sampled requests).

Registration: vllm_bundle/hy4_flash/vllm_metadata.json (EXTRA_MODELS_DIR), the HF config / `hy_v4` parsers through
serving/sitecustomize.py.
"""

from __future__ import annotations

import math
import os

import torch
from loguru import logger

HF_MODEL = "tencent/Hy4-flash"
BLOCK_SIZE = 64
SPEC_K = int(os.environ.get("HY4_SPEC_K", "0"))


def _mtp_sampling(sp, row):
    """(temperature, top_k, top_p, seed) of request `row` from the plugin's sampling params; None when absent."""
    if sp is None:
        return None

    def pick(v, default):
        if v is None:
            return default
        if isinstance(v, (int, float)):
            return v
        try:
            x = v[row]
        except (IndexError, TypeError):
            return default
        if isinstance(x, torch.Tensor):
            x = x.item()
        return default if x is None else x

    t = float(pick(getattr(sp, "temperature", None), 0.0))
    k = int(pick(getattr(sp, "top_k", None), 0) or 0)
    p = float(pick(getattr(sp, "top_p", None), 1.0))
    seed = pick(getattr(sp, "seed", None), None)
    seed = None if seed is None or int(seed) < 0 else int(seed)
    return (t, k, p, seed)


class HYV4FlashForConditionalGeneration:
    """vLLM bridge (HF arch HYV4FlashForConditionalGeneration; text only)."""

    supports_multimodal = False
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_sample_on_device": True,
        "supports_chunked_prefill": False,
    }
    decode_input_update_contract = 1

    def __init__(self, model, engine, mesh_device, max_seq_len):
        self.model, self._engine, self.mesh_device = model, engine, mesh_device
        self._context = max_seq_len
        self._warmed = False
        self.already_warmed_up_prefill = False
        self._decode_steps = 0

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len=None, tt_data_parallel=1,
                              optimizations=None, **kwargs):
        if int(tt_data_parallel) != 1 or int(max_batch_size) != 1:
            raise ValueError(f"{cls.__name__} serves one user on one (1,4) mesh (--max-num-seqs 1)")
        max_seq_len = int(max_seq_len or 131072)
        from models.autoports.tencent_hy4_flash.tt.config import Hy4Config, checkpoint_dir
        from models.autoports.tencent_hy4_flash.tt.engine import Hy4Engine
        from models.autoports.tencent_hy4_flash.tt.model import Hy4Model
        from models.experimental.hybrid_moe.engram_store import EngramRows

        cfg = Hy4Config.load()
        engram = EngramRows(os.environ.get("HY4_ENGRAM_DIR", os.path.expanduser("~/.cache/hybrid-moe/hy4_engram")),
                             checkpoint_dir())
        cache_dir = os.environ.get("HY4_TENSOR_CACHE", os.path.expanduser("~/.cache/hybrid-moe/hy4_tt"))
        model = Hy4Model(mesh_device, cfg, max_ctx=max_seq_len + 64, chunk=int(os.environ.get("HY4_PREFILL_CHUNK", "2048")),
                         engram=engram, cache_dir=cache_dir)
        engine = Hy4Engine(model, SPEC_K)
        logger.info(f"{cls.__name__}: max_model_len={max_seq_len} layers={len(model.layers)} spec k={SPEC_K}")
        return cls(model, engine, mesh_device, max_seq_len)

    # ------------------------------------------------------------------ KV: a token budget only
    @classmethod
    def get_max_tokens_all_users(cls, model_name="", num_devices=1, tt_data_parallel=1, max_model_len=None,
                                 max_num_seqs=None, **kwargs):
        n = int(max_model_len or 131072) + BLOCK_SIZE
        return math.ceil(n / BLOCK_SIZE) * BLOCK_SIZE

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        logger.info(f"KV: vLLM asked for {num_layers} layers of {tuple(kv_cache_shape)}; the engine keeps its own caches")
        return [(None, None)] * int(num_layers)

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        return [(None, None)] * len(per_layer_specs)

    # ------------------------------------------------------------------ warm-up
    def warmup_model_prefill(self, kv_cache, enable_trace, *args, **kwargs):
        if not self._warmed:
            self._engine.warm()
            self._warmed = True
        self.already_warmed_up_prefill = True

    def warmup_model_decode(self, *args, **kwargs):
        if not self._warmed:
            self._engine.warm()
            self._warmed = True
        if kwargs.get("enable_trace", False):
            self._engine.capture()

    # ------------------------------------------------------------------ prefill
    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        N = int(tokens.shape[0])
        start_pos = kwargs.get("start_pos")
        if start_pos is not None and any(int(v) for v in torch.as_tensor(start_pos).reshape(-1).tolist()):
            raise NotImplementedError("prefix caching / chunked prefill are off for Hy4 (start_pos must be 0)")
        plens = [int(prompt_lens[u]) for u in range(N)] if prompt_lens is not None else [int(tokens.shape[1])] * N
        sp = kwargs.get("sampling_params")
        out = []
        for u in range(N):
            out.append(self._engine.prefill(tokens[u, : plens[u]].tolist(), _mtp_sampling(sp, u)))
        if sp is not None:
            return torch.tensor(out, dtype=torch.int64).reshape(N, 1)
        self._engine.adopt_next = True
        V = int(self._engine.V)
        logits = torch.full((N, 1, V), float("-inf"), dtype=torch.float32)
        logits[:, 0, :] = self._engine.last_logits[:V]
        return logits

    # ------------------------------------------------------------------ decode
    def decode_forward(self, *args, **kwargs):
        tokens = torch.as_tensor(kwargs.get("tokens", args[0] if args else None)).reshape(-1)
        pos = torch.as_tensor(kwargs.get("start_pos", args[1] if len(args) > 1 else None)).reshape(-1)
        sp = kwargs.get("sampling_params", args[6] if len(args) > 6 else None)
        live = [i for i in range(pos.numel()) if int(pos[i]) >= 0]
        assert len(live) <= 1, "single-user Hy4 step got several live rows"
        self._decode_steps += 1
        if sp is None:  # host sampling: next-token logits
            V = int(self._engine.V)
            logits = torch.full((tokens.numel(), 1, V), float("-inf"), dtype=torch.float32)
            for i in live:
                logits[i, 0, :] = self._engine.step_logits(int(tokens[i]), int(pos[i]))[:V]
            return logits
        out = torch.full((tokens.numel(), 1), -1, dtype=torch.int32)
        authoritative = bool(kwargs.get("reload_inputs", False) or kwargs.get("reset_batch", False))
        for i in live:
            ids = self._engine.step(int(tokens[i]), int(pos[i]), host_authoritative=authoritative,
                                    sampling=_mtp_sampling(sp, i))
            out[i, : len(ids)] = torch.tensor(ids[:1], dtype=torch.int32)
        if self._decode_steps % 1024 == 0:
            logger.info(f"Hy4 engine stats: {self._engine.stats}")
        return out

    def read_decode_output(self, tt_out, async_read=False):
        return (tt_out, []) if async_read else tt_out

    def process_decode_output_host(self, tt_out, is_tokens=True):
        return tt_out


__all__ = ["HYV4FlashForConditionalGeneration", "HF_MODEL"]
