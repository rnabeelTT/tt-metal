# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full Qwen/Qwen3.5-122B-A10B text model on the (1,4) Blackhole mesh (stage 6).

``Qwen35MoeModel`` is the qwen36 hybrid full model (embeddings, 48 decoder layers, final norm, vocab-sharded LM head,
on-device sampling) with the stage-5 optimized multichip MoE decoder layer as ``DECODER_LAYER_CLS``. The policy from
``MULTICHIP_SELECTED`` is installed on the args once (``MultichipDecoder._args_hook``) and every layer shares the model's
``TT_CCL``. Weights stream per layer from the safetensors shards (``StreamingStateDict.layer``); converted tensors are
cached under ``Qwen36ModelArgs.weight_cache_path()`` (mesh-qualified).

Decode trace contract (canonical split sampling, `$tt-enable-tracing`): the persistent decode inputs are
``(tokens uint32 [B,1], current_pos int32 [B], rot_idxs uint32 [1,B], page_table int32 [B,blocks])``. RoPE cos/sin are
gathered ON DEVICE from resident tables by ``rot_idxs`` (no host cos/sin per token), and when the on-device sampler
consumes the logits the forward advances ``current_pos`` and ``rot_idxs`` with ``ttnn.plus_one`` inside the trace, so the
sampled token written into ``tokens`` by ``tt_out_tok`` and the positions stay coherent without any host refresh.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tt.model_config import Qwen35MoeModelArgs
from models.autoports.qwen_qwen3_5_122b_a10b.tt.multichip_decoder import (
    MULTICHIP_SELECTED,
    MultichipConfig,
    MultichipDecoder,
)
from models.autoports.qwen_qwen3_5_122b_a10b.tt.optimized_decoder import Qwen35MoeOptimizedDecoderLayer
from models.demos.blackhole.qwen36.tt.generator_interface import unpack_rope
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.tt_transformers.tt.common import Mode


class _CacheRoot(type(Path())):
    """Weight-cache root handed to the qwen36 / tt_transformers loaders, which build cache names with ``root / key``.
    ``disabled=True`` turns every key into ``None`` (no tensor file cache): ``ttnn.load_tensor`` of this model's cached
    multi-device weights never returns within minutes on this host (stage-6 work_log 16:05), while a fresh conversion from
    the streaming checkpoint is fast. The framework ``Embedding`` requires a root object even when caching is off."""

    _disabled = False

    def __new__(cls, *args, disabled=False, **kwargs):
        self = super().__new__(cls, *args, **kwargs)
        self._disabled = disabled
        return self

    def __bool__(self):
        # callers gate their cache use on truthiness (``... if tensor_cache_path else None``); a disabled root is falsy
        return not self._disabled

    def __truediv__(self, key):
        if self._disabled or "tok_embeddings" in str(key):
            return None
        return super().__truediv__(key)


class Qwen35MoeModel(Qwen36Model):
    """qwen36 full model with the optimized multichip MoE layer; see the module docstring for the decode contract."""

    DECODER_LAYER_CLS = Qwen35MoeOptimizedDecoderLayer

    def __init__(self, mesh_device, args, state_dict, tensor_cache_path=None):
        if not isinstance(tensor_cache_path, _CacheRoot):
            tensor_cache_path = _CacheRoot(tensor_cache_path or ".", disabled=tensor_cache_path is None)
        super().__init__(mesh_device, args, state_dict, tensor_cache_path)
        # Device-owned decode inputs: tokens via tt_out_tok, positions via plus_one (no per-token host refresh).
        self._tt_vllm_always_refresh_decode_trace_inputs = False
        rd = self.args.rope_head_dim
        rep = ttnn.ReplicateTensorToMesh(mesh_device) if self.num_devices > 1 else None
        kw = {"mesh_mapper": rep} if rep is not None else {}
        # 2-D ROW_MAJOR tables for the ttnn.embedding gather (the base class keeps 3-D TILE tables for slicing).
        self._rope_cos_2d = ttnn.from_torch(
            self.rope.cos_cpu.reshape(-1, rd).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            **kw,
        )
        self._rope_sin_2d = ttnn.from_torch(
            self.rope.sin_cpu.reshape(-1, rd).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            **kw,
        )

    # ---- weights -------------------------------------------------------------------------------------------------
    @staticmethod
    def _layer_state_dict(state_dict, layer_idx):
        """One layer's tensors from the streaming checkpoint (``substate`` over the whole mapping would read 250 GB)."""
        if hasattr(state_dict, "layer"):
            return state_dict.layer(layer_idx)
        return state_dict

    # ---- decode: device-resident positions + RoPE -----------------------------------------------------------------
    def prepare_decode_inputs_host(self, tokens, current_pos, page_table=None):
        """HOST decode inputs: (tokens_tt uint32 [1,1,1,S], cur_pos_tt int32 [B], rot_idxs uint32 [1,B], page_table_tt).

        The token buffer has the SAMPLER's batch shape (S = the sampler's padded batch, 32) so the on-device sampler can
        write the sampled tokens straight into it (`ttnn.sampling` needs a rank-4 [1,1,1,S] output; `ttnn.argmax` accepts
        it too); the decode step reads its B tokens back out of it (`ttnn_decode_forward`). Rows >= B are padding."""
        B = tokens.shape[0]
        S = self._token_buffer_rows()
        tok = torch.zeros(1, 1, 1, S, dtype=torch.int32)
        tok[0, 0, 0, :B] = tokens.reshape(-1).to(torch.int32)
        tokens_tt = ttnn.from_torch(tok, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        if isinstance(current_pos, torch.Tensor):
            pos_vec = current_pos.to(torch.int32).reshape(-1)
            assert pos_vec.shape[0] == B, f"current_pos length {pos_vec.shape[0]} != batch {B}"
        else:
            pos_vec = torch.full((B,), int(current_pos), dtype=torch.int32)
        rope_pos = (pos_vec + self.rope.rope_delta).clamp(min=0).to(torch.int32)
        cur_pos_tt = ttnn.from_torch(pos_vec, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        rot_idxs = ttnn.from_torch(rope_pos.reshape(1, B), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        page_table_tt = (
            ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            if page_table is not None
            else None
        )
        return tokens_tt, cur_pos_tt, rot_idxs, page_table_tt

    def _token_buffer_rows(self):
        """Rows of the persistent decode token buffer = the sampler's padded batch (32), else the model batch."""
        if self.sampling is not None:
            return int(self.sampling.tt_sampling.max_batch_size)
        return int(self.args.max_batch_size)

    def _decode_tokens_view(self, tokens, B):
        """The B decode tokens ([B,1] uint32) out of the persistent [1,1,1,S] buffer (a legacy [B,1] buffer passes through)."""
        if len(tokens.shape) == 4:
            S = tokens.shape[-1]
            view = ttnn.reshape(tokens, (S, 1))  # row-major view of the same buffer
            return view if S == B else ttnn.slice(view, (0, 0), (B, 1))
        return tokens

    def _rope_from_idxs(self, rot_idxs):
        """cos/sin [1,B,1,rope_dim] TILE gathered on device from the resident tables."""
        B = rot_idxs.shape[-1]
        cos = ttnn.embedding(
            rot_idxs, self._rope_cos_2d, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        sin = ttnn.embedding(
            rot_idxs, self._rope_sin_2d, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        return ttnn.reshape(cos, (1, B, 1, cos.shape[-1])), ttnn.reshape(sin, (1, B, 1, sin.shape[-1]))

    def ttnn_decode_forward(
        self, tokens, current_pos, rot_mat_idxs=None, page_table=None, kv_cache=None, on_device_logits=False, **kwargs
    ):
        """Trace-safe decode step over the persistent device inputs (see module docstring)."""
        if rot_mat_idxs is None or len(rot_mat_idxs.shape) == 4:
            # legacy packed host cos/sin (tests, eager helpers)
            cos, sin = unpack_rope(rot_mat_idxs) if rot_mat_idxs is not None else (None, None)
        else:
            cos, sin = self._rope_from_idxs(rot_mat_idxs)
        B = current_pos.shape[-1]
        tokens = self._decode_tokens_view(tokens, B)
        if on_device_logits:
            assert self.sampling is not None, "on_device_logits=True but self.sampling is None"
            logits = self._forward_decode(tokens, cos, sin, current_pos, page_table, sharded_lm_head=True)
            sampler_batch = self.sampling.tt_sampling.max_batch_size
            B = logits.shape[2]
            if B < sampler_batch:
                logits = ttnn.pad(logits, [(0, 0), (0, 0), (0, sampler_batch - B), (0, 0)], value=0.0)
            # device-side position advance for the token-feedback loop (host inputs are only re-written when the
            # caller is authoritative: first step after prefill, teacher forcing, page-table changes)
            ttnn.plus_one(current_pos, skip_negative_entries=True)
            if rot_mat_idxs is not None and len(rot_mat_idxs.shape) != 4:
                ttnn.plus_one(rot_mat_idxs)
            return logits
        logits = self._forward_decode(tokens, cos, sin, current_pos, page_table)
        return logits, None

    # ---- prefill: logits at every position (readiness prefill check) ---------------------------------------------
    def prefill_all_logits(self, token_ids, page_table, valid_len=None):
        """B=1 paged prefill returning host logits ``[1, T, vocab]`` for EVERY position (fills KV + GDN state as the
        regular prefill does). ``valid_len`` < T masks the tail like ``_prefill_paged_tp``."""
        B, T = token_ids.shape
        assert B == 1
        vlen = valid_len or T
        # the TP GDN prefill kernels take whole 128-token chunks: pad the tokens, mask with valid_len, slice the logits
        chunk = int(self.args.gdn_chunk_size)
        T_pad = -(-T // chunk) * chunk
        if T_pad != T:
            pad_id = getattr(self.args, "pad_token_id", None) or 0
            token_ids = torch.cat([token_ids, torch.full((1, T_pad - T), int(pad_id), dtype=token_ids.dtype)], dim=1)
            T = T_pad
        self._build_request_rope(token_ids[:, :vlen], None)
        pt_torch = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        page_table_tt = ttnn.from_torch(pt_torch, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        tok = ttnn.from_torch(
            token_ids.to(torch.int32),
            dtype=ttnn.uint32,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        x = self.embd(tok)
        x = ttnn.reshape(x, (1, 1, T, x.shape[-1]))
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        cos_t, sin_t = self._rope_tp_cos_sin_torch(0, T)
        rep = ttnn.ReplicateTensorToMesh(self.device)
        cos = ttnn.from_torch(cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=rep)
        sin = ttnn.from_torch(sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=rep)
        for layer in self.layers:
            x = layer.forward(
                x,
                cos=cos,
                sin=sin,
                mode="prefill",
                chunk_size=self.args.gdn_chunk_size,
                valid_len=vlen,
                page_table=page_table_tt,
                chunk_page_table=page_table_tt,
                chunk_start_idx=0,
            )
        x = self.norm(x, mode=Mode.PREFILL)
        logits = self._lm_head(x)  # replicated [1,1,T,vocab]
        ttnn.deallocate(x)
        out = ttnn.to_torch(ttnn.get_device_tensors(logits)[0]).float().reshape(1, -1, self.args.vocab_size)[:, :vlen]
        ttnn.deallocate(logits)
        return out


def create_model(
    mesh_device,
    *,
    max_batch_size=1,
    max_seq_len=None,
    n_layers=None,
    layer_indices=None,
    mc: MultichipConfig | None = None,
):
    """Build (args, model) for the mesh. ``layer_indices`` (checkpoint indices) selects a reduced model for probes/profiling."""
    mc = mc or MULTICHIP_SELECTED
    if os.environ.get("QWEN35_MC_OVERRIDE"):  # experiments: '{"gdn_prefill_mmrs": false}'
        import json

        mc = mc.with_(
            **{
                k: (tuple(v) if isinstance(v, list) else v)
                for k, v in json.loads(os.environ["QWEN35_MC_OVERRIDE"]).items()
            }
        )
    if tuple(mesh_device.shape) != tuple(mc.mesh_shape):
        raise ValueError(f"mesh {tuple(mesh_device.shape)} != MultichipConfig.mesh_shape {tuple(mc.mesh_shape)}")
    args = Qwen35MoeModelArgs(mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len or 262144)
    if max_seq_len is None:
        args.max_seq_len = args.hf_max_context
    MultichipDecoder._args_hook(args, mc, mc.to_optimization_config(), mesh_device)
    # Greedy split-sampling strategy of the common sampler: force-argmax (one all-gather of the full vocab + argmax, the
    # qwen36 default on this mesh) vs the top-k candidate path (per-shard top-k, all-gather of 32 candidates, ttnn.sampling
    # with k=1). QWEN35_FORCE_ARGMAX=0/1 overrides for the A/B measurement (doc/full_model/README.md section 3).
    fa = os.environ.get(
        "QWEN35_FORCE_ARGMAX", "0"
    )  # default: the top-k candidate path (stage-6 measurement, README section 3)
    if "SAMPLING_AG_CONFIG" in getattr(args, "model_config", {}):
        args.model_config["SAMPLING_AG_CONFIG"]["allow_force_argmax"] = fa == "1"
    if layer_indices is not None:
        layer_indices = [int(i) for i in layer_indices]
        assert layer_indices and all(0 <= i < len(args.attention_type_list) for i in layer_indices), layer_indices
        args.layer_indices = layer_indices
        args.n_layers = len(layer_indices)
    elif n_layers is not None:
        args.n_layers = int(n_layers)
        args.attention_type_list = args.attention_type_list[: int(n_layers)]
    # No ttnn tensor-file cache by default: ``ttnn.load_tensor`` of this model's cached multi-device weights (67 MB TILE
    # bfp8, 1.5 GB ROW_MAJOR bf16) never returns within minutes on this host (stage-6 work_log 16:05; the write is fast),
    # while converting a layer from the streaming safetensors shards takes ~11 s (bfp4 expert packing included). Set
    # QWEN35_TENSOR_CACHE=1 to opt back into the cache for experiments.
    use_cache = os.environ.get("QWEN35_TENSOR_CACHE") == "1"
    cache_path = _CacheRoot(args.weight_cache_path(), disabled=not use_cache)
    state_dict = args.load_state_dict()
    model = Qwen35MoeModel(mesh_device, args, state_dict, tensor_cache_path=cache_path)
    model.mc = mc
    return args, model
