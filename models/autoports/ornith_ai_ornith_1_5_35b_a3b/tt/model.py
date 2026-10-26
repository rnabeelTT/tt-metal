# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full ornith-ai/Ornith-1.5-35B-A3B text model on the (1,4) Blackhole mesh (stage 6).

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

import json
import os
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt import precision
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.model_config import (
    Qwen35MoeModelArgs,
    check_mesh_shape,
    mesh_shape_from_env,
)
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.multichip_decoder import (
    MULTICHIP_SELECTED,
    MultichipConfig,
    MultichipDecoder,
)
from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.optimized_decoder import Qwen35MoeOptimizedDecoderLayer
from models.common.sampling.generator import SeedManager
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


class _LockstepSeedManager(SeedManager):
    """SeedManager whose UNSEEDED path pushes fresh per-slot seeds before EVERY draw.

    The stock manager pushes concrete seeds once and then MAX_UINT32 ("SKIP"), which leaves each die's hardware PRNG
    free-running. ``ttnn.sampling`` runs on EVERY die of the mesh (each die holds the same gathered candidate row), the
    host reads die 0's token (``process_output_decode``) and every die feeds ITS OWN token back into the next decode
    step (``tt_out_tok`` is the persistent decode token input). The free-running PRNGs stay in lockstep only while every
    die advances its PRNG identically between draws. Inside this model's decode step they do not (with only the sampler
    running the dies stay in lockstep; which decode kernel desynchronises them is not pinned down), so the dies drift
    apart and draw DIFFERENT tokens from the same distribution (measured on p300x2 at 20K context: 196 of 400 sampled
    steps had at least one die disagree with die 0). Die 0's token then goes to the user while the model
    continues from the other dies' tokens: the long-context word corruption of doc/text_quality/README.md
    ("checkpoints" + "etensors", "interpreterterpreter").

    Identical seeds pushed right before each draw (``ttnn.manual_seed`` runs immediately before ``ttnn.sampling`` in the
    sampler trace, the seed tensor is replicated on the mesh) re-synchronise every die on every step: 0 % cross-die
    disagreement and ~88 % step-to-step draw variation in the standalone 4-die op test. The seeded path is unchanged
    (it already pushes concrete per-step seeds for every live slot). Cost: one 128-byte host-to-device copy per step.
    """

    def get_new_values(self, empty_slots=None, replicate_seeds=False):
        if self._seed_active:
            return super().get_new_values(empty_slots, replicate_seeds=replicate_seeds)
        self._active_request_seed = False
        new_seeds = [self._next_unseeded_device_seed() for _ in range(self.max_batch_size)]
        self.write_device_seed_values(new_seeds)
        self._reseted = False
        self._needs_skip = False
        return tuple(new_seeds)


def _install_lockstep_sampling(sampling, mesh_device):
    """Make every die of the mesh feed back the SAME sampled token the host reads (see _LockstepSeedManager).

    Two independent guarantees: (1) identical fresh seeds before every draw, so the dies draw identically; (2) after
    ``ttnn.sampling`` the token row is all-gathered and die 0's row is copied into the token buffer on every die, so
    even a die whose draw differed for any other reason feeds back die 0's token - the one the host reports.
    QWEN35_SAMPLER_TOKEN_BROADCAST=0 disables (2) (A/B only). The greedy force-argmax path (not enabled on this mesh)
    is deterministic per die and is left alone.
    """
    old = sampling.seed_manager
    sampling.seed_manager = _LockstepSeedManager(
        sampling.tt_sampling, max_batch_size=old.max_batch_size, salt_duplicate_seeds=old.salt_duplicate_seeds
    )
    ts = sampling.tt_sampling
    orig_forward = ts.forward
    broadcast = os.environ.get("QWEN35_SAMPLER_TOKEN_BROADCAST", "1") == "1"

    def forward(x, tt_out_tok=None):
        tokens, log_probs = orig_forward(x, tt_out_tok=tt_out_tok)
        if not broadcast or ts.force_argmax_sampling:
            return tokens, log_probs
        # [1,1,1,S] uint32 ROW_MAJOR on every die -> [1,1,N,S] -> die 0's row -> back into the (persistent) buffer.
        # (The sampled-token log-probs, when enabled, were computed from each die's own draw; on this mesh the plugin
        # samples on the host whenever log-probs are requested, so that tensor is never consumed here.)
        gathered = ttnn.all_gather(tokens, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        first = ttnn.slice(gathered, [0, 0, 0, 0], [1, 1, 1, tokens.shape[-1]])
        ttnn.deallocate(gathered)
        ttnn.copy(first, tokens)
        ttnn.deallocate(first)
        return tokens, log_probs

    ts.forward = forward
    logger.info(
        f"sampler lockstep installed on {mesh_device.get_num_devices()} dies: fresh identical seeds every draw"
        f"{', die-0 token broadcast' if broadcast else ''}"
    )


def _install_rows_topk(sampling, mesh_device):
    """QWEN35_ROWS_TOPK=1: the sampler's per-device vocab mask + power-of-two pad + ttnn.topk over all 32 tile rows (333 us
    per decode step on the (1, 4) mesh) -> decode_sampling.RowsTopK over the live rows only, for steps of <= MAX_ROWS rows
    when every vocab column is valid (no mask needed). The rest of the sampler (all-gathers, offsets, draw) is unchanged:
    its forward runs with ttnn.topk pointed at RowsTopK for the duration of the call."""
    from models.demos.blackhole.qwen36.tt.decode_sampling.ops import MAX_ROWS, MAX_WIDTH, RowsTopK

    ts = sampling.tt_sampling
    nd = mesh_device.get_num_devices()
    if nd < 2 or ts.padded_vocab_size != ts.vocab_size or ts.vocab_size % nd or ts.max_top_k != 32:
        logger.info("rows top-k not installed (padded vocab or k != 32)")
        return
    if ts.vocab_size // nd > MAX_WIDTH:  # 16-bit indices: Ornith / 122B on 2 dies (124160 columns) keep ttnn.topk
        logger.info(f"rows top-k not installed ({ts.vocab_size // nd} columns per device > {MAX_WIDTH})")
        return
    op = RowsTopK(mesh_device, ts.vocab_size // nd, 32)
    orig_forward = ts.forward

    seen = set()
    force_rows = int(os.environ.get("QWEN35_ROWS_TOPK_ROWS", "0"))  # tests without decode buckets: the live row count

    def forward(x, tt_out_tok=None):
        # the logits always carry 32 slot rows; the live ones are the decode bucket's width (the served path traces the
        # sampler per width, SamplingGenerator.set_trace_bucket); no bucket (eager / tests): all 32
        B = getattr(sampling, "_active_trace_bucket", None) or force_rows or int(x.shape[-2])
        if ts.force_argmax_sampling or B > MAX_ROWS or getattr(ts, "multi_step_reduction", False):
            return orig_forward(x, tt_out_tok=tt_out_tok)
        if B not in seen:
            seen.add(B)
            logger.info(f"rows top-k: sampler step with {B} live rows (logits {tuple(x.shape)})")
        # RowsTopK emits tied maxima in ascending local index and the gathered blocks are in device order, so candidate
        # position and global id agree and ttnn.sampling's first-position tie-break already picks the lowest id: the
        # sampler's tie-break pass (the #33492 workaround for ttnn.topk, ~15 ops) is skipped
        saved = (ttnn.topk, ts._mask_invalid_vocab_logits, ts.pad_to_power_of_2, ts._adjust_values_for_tiebreak)
        ttnn.topk = lambda xx, k, dim=-1, sub_core_grids=None, stable=False: op(xx, B)
        ts._mask_invalid_vocab_logits = lambda xx: xx
        ts.pad_to_power_of_2 = False
        ts._adjust_values_for_tiebreak = lambda values, indices: ttnn.clone(values)
        try:
            return orig_forward(x, tt_out_tok=tt_out_tok)
        finally:
            ttnn.topk, ts._mask_invalid_vocab_logits, ts.pad_to_power_of_2, ts._adjust_values_for_tiebreak = saved

    ts.forward = forward
    logger.info(f"sampler rows top-k installed (<= {MAX_ROWS} rows, {ts.vocab_size // nd} columns per die)")


# module scope: a class-body generator expression cannot see class attributes
_PREFILL_SEGMENT = int(os.environ.get("QWEN35_PREFILL_SEGMENT", "1024" if mesh_shape_from_env() == (1, 1) else "2048"))


class Qwen35MoeModel(Qwen36Model):
    """qwen36 full model with the optimized multichip MoE layer; see the module docstring for the decode contract."""

    # Prefill segment length (perf pass turn 5): the segmented prefill and the served single-request prefill run one
    # bucket program per PREFILL_SEGMENT tokens (stages 6-13: 2048). QWEN35_PREFILL_SEGMENT=4096 adds a 4096-token
    # bucket (expert-sorted dispatch is exact to 7196 tokens; needs a larger trace region): fewer, fuller segments
    # amortize the per-segment dispatch bookkeeping of the MoE block.
    # One die (MESH_DEVICE=P150): 1024 - the full-width GDN prefill programs of a 2048-token bucket do not fit L1
    # there (static circular buffers clash with L1 buffers); 1024-token segments carry the state exactly as 2048 do.
    PREFILL_SEGMENT = _PREFILL_SEGMENT
    _PREFILL_MASK_BUCKETS = tuple(
        b for b in (128, 256, 512, 1024, 2048, 4096, 8192) if b <= max(1024, _PREFILL_SEGMENT)
    )

    DECODER_LAYER_CLS = Qwen35MoeOptimizedDecoderLayer

    def __init__(self, mesh_device, args, state_dict, tensor_cache_path=None):
        if not isinstance(tensor_cache_path, _CacheRoot):
            tensor_cache_path = _CacheRoot(tensor_cache_path or ".", disabled=tensor_cache_path is None)
        super().__init__(mesh_device, args, state_dict, tensor_cache_path)
        if self.sampling is None and tuple(mesh_device.shape) == (1, 2) and args.vocab_size % 2 == 0:
            # TP2 (P300 / P150x2): the qwen36 allowlist predates Blackhole's routed top-k. The per-device shard
            # (151936 / 2 = 75968, padded to 131072) exceeds the stock 64K top-k width, but on Blackhole ttnn.topk
            # routes such rows to topk_large_indices (up to 2^19 columns, uint32 indices), which the common sampler's
            # multi-device path already consumes.
            from models.common.sampling.generator import SamplingGenerator

            args.pad_logits_to_power_of_2 = True
            self._supports_on_device_sampling = True
            self.sampling = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=self.tt_ccl)
        elif self.sampling is None and tuple(mesh_device.shape) == (1, 1):
            # one die (P150): the common sampler's single-device multi-step reduction (the full vocab row in power-of-two
            # tile-aligned chunks that each fit ttnn.topk, or Blackhole's topk_large route); no logits read-back per step
            from models.common.sampling.generator import SamplingGenerator
            from models.common.sampling.tt_sampling import TTSampling

            if (
                TTSampling.num_single_device_vocab_splits(getattr(args, "padded_vocab_size", args.vocab_size))
                is not None
            ):
                self._supports_on_device_sampling = True
                self.sampling = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=self.tt_ccl)
        if (
            self.sampling is not None
            and mesh_device.get_num_devices() > 1
            and os.environ.get("QWEN35_SAMPLER_LOCKSTEP", "1") == "1"
        ):
            # every die must feed back the token the host reads (doc/text_quality/README.md root cause);
            # QWEN35_SAMPLER_LOCKSTEP=0 restores the stock sampler (A/B measurement only)
            if os.environ.get("QWEN35_ROWS_TOPK", "0") == "1":
                _install_rows_topk(self.sampling, mesh_device)
            _install_lockstep_sampling(self.sampling, mesh_device)
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
        # multimodal requests decode at rope position = KV position + their M-RoPE delta; a multi-slot server keeps one
        # delta per slot (set after each request's prefill, `set_slot_rope_delta`), text requests have delta 0
        slot_delta = self.__dict__.get("_slot_rope_delta")
        if slot_delta is not None and B <= slot_delta.numel():
            rope_pos = (pos_vec + slot_delta[:B]).clamp(min=0).to(torch.int32)
        else:
            rope_pos = (pos_vec + self.rope.rope_delta).clamp(min=0).to(torch.int32)
        cur_pos_tt = ttnn.from_torch(pos_vec, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        rot_idxs = ttnn.from_torch(rope_pos.reshape(1, B), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        page_table_tt = (
            ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            if page_table is not None
            else None
        )
        return tokens_tt, cur_pos_tt, rot_idxs, page_table_tt

    def set_slot_rope_delta(self, slot, delta):
        """Record decode slot ``slot``'s M-RoPE position delta (0 for text requests)."""
        vec = self.__dict__.get("_slot_rope_delta")
        if vec is None:
            vec = torch.zeros(int(self.args.max_batch_size), dtype=torch.int32)
            self._slot_rope_delta = vec
        vec[int(slot)] = int(delta)

    def slot_rope_delta(self, slot):
        vec = self.__dict__.get("_slot_rope_delta")
        return int(vec[int(slot)]) if vec is not None else 0

    def _token_buffer_rows(self):
        """Rows of the persistent decode token buffer = the sampler's padded batch (32), else the model batch."""
        if self.sampling is not None:
            return int(self.sampling.tt_sampling.max_batch_size)
        return int(self.args.max_batch_size)

    def _snapshot_gdn_scratch_device(self):
        """Device clones of the B=1 GDN prefill scratch (recurrent state + the K conv taps of every GDN layer) - the
        request's state right after its prefill, while the scratch is bound. ~1 MiB per layer per device; consumed by
        :meth:`_write_gdn_slot_device`."""
        dn_layers = [layer.attention for layer in self.layers if not layer.is_full_attention]
        return (
            [ttnn.clone(dn.rec_state) for dn in dn_layers],
            [[ttnn.clone(c) for c in dn.conv_states] for dn in dn_layers],
        )

    def _write_gdn_slot_device(self, slot, recs, convs):
        """Write a device snapshot (:meth:`_snapshot_gdn_scratch_device`) into decode ``slot`` of the batched GDN buffers
        on device (``TPGatedDeltaNet.write_slot``: slice / concat / in-place copy; consumes the snapshot). Stage 10: replaces
        the per-request host round trip (to_torch of 36 layers' states + from_torch uploads, ~80 MB, ~120 ms of served TTFT
        at 32 slots) of the qwen36 ``_write_gdn_slot``."""
        dn_layers = [layer.attention for layer in self.layers if not layer.is_full_attention]
        assert len(recs) == len(dn_layers) == len(convs)
        for dn, rec, conv in zip(dn_layers, recs, convs):
            dn.write_slot(slot, rec, conv)

    def _decode_tokens_view(self, tokens, B):
        """The B decode tokens out of the persistent [1,1,1,S] buffer as a ``[1, B]`` uint32 row (a legacy [B,1] buffer
        passes through). Stage 10: the row layout makes the embedding emit ``[1, B, dim]``, so the decode step's
        ``reshape`` to ``[1, 1, B, dim]`` is a free view at EVERY width; the former ``[B, 1]`` layout gave ``[B, 1, dim]``
        (one tile row per token) and a real tiled re-pack for 1 < B < 32 - the kernel that hung the width probe
        (``doc/optimized_vllm/triage/width2_reshape_hang_triage.txt``: reader_reshape_tiled in cb_reserve_back)."""
        if len(tokens.shape) == 4:
            S = tokens.shape[-1]
            view = ttnn.reshape(tokens, (1, S))  # row-major view of the same buffer
            return view if S == B else ttnn.slice(view, (0, 0), (1, B))
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

    def _forward_decode(self, token_ids_buf, cos, sin, cur_pos_tensor, page_table, sharded_lm_head=False):
        """The base decode body with the adaptive-MTP hooks (`_mtp_decode_hook`, set by MtpServing): before the layers
        the head's K/V rows of (this step's tokens, the previous step's final hidden), after the final norm the hidden
        rows for the next step. Both run inside the decode traces; no-ops without MTP."""
        hook = self.__dict__.get("_mtp_decode_hook")
        x = self.embd(token_ids_buf)
        if self.num_devices > 1 or getattr(self.args, "tp_modules_single_device", False):
            x = ttnn.reshape(x, (1, 1, x.shape[0] * x.shape[1], x.shape[-1]))
        replicated = bool(getattr(self.args, "decode_replicated", False))
        if replicated:  # the replicated decode residual (decode_boundary.py): one gather of the embedding rows
            from models.demos.blackhole.qwen36.tt.decode_boundary import decode_boundary

            bnd = decode_boundary(self.args, self.mesh_device)
            x = bnd.gather(x, self.tt_ccl)
        if hook is None and x.shape[-2] <= getattr(self.args, "fabric_boundary_rows", 1) and getattr(
            self.args, "fabric_boundary", False
        ):
            x = self._decode_layers_fabric(x, cos, sin, cur_pos_tensor, page_table)
            if os.environ.get("QWEN35_LM_STREAM", "0") == "1" and (sharded_lm_head or self._ondev_argmax):
                # the LM head streamed per DRAM bank (decode_linear/ops.py; its decode copy built on the first, eager,
                # call): 261 us at ~274 GB/s with ttnn.linear (bfp4, 71.5 MB per device)
                if getattr(self, "_lm_stream", None) is None:
                    from models.demos.blackhole.qwen36.tt.decode_linear.ops import decode_linear_for

                    self._lm_stream = decode_linear_for(self.args, self.mesh_device, self.lm_head_weight, readers=6)
                op, w = self._lm_stream
                logits = op(x, w, int(x.shape[-2]), out_memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(x)
                return logits
            if sharded_lm_head or self._ondev_argmax:
                logits = ttnn.linear(x, self.lm_head_weight, compute_kernel_config=self._lm_head_ckc())
            else:
                logits = self._lm_head(x)
            ttnn.deallocate(x)
            return logits
        if hook is not None:
            hook("pre", token_ids_buf, cur_pos_tensor, page_table, cos, sin)
        for layer in self.layers:
            if layer.is_full_attention:
                x = layer.forward(x, cos, sin, position_tensor=cur_pos_tensor, page_table=page_table, mode="decode")
            else:
                x = layer.forward(x, mode="decode")
        x = self._final_norm_decode_replicated(x) if replicated else self._final_norm_decode(x)
        if hook is not None:
            hook("post", x)
        if sharded_lm_head or self._ondev_argmax:
            logits = ttnn.linear(x, self.lm_head_weight, compute_kernel_config=self._lm_head_ckc())
        else:
            logits = self._lm_head(x)
        ttnn.deallocate(x)
        return logits

    def _decode_layers_fabric(self, x, cos, sin, cur_pos_tensor, page_table):
        """One-row decode through the fabric boundaries (args.fabric_boundary): x is the gathered embedding row in the
        replicated residual layout; the entry boundary applies layer 0's input norm, every layer ends with the boundary
        into the next layer's input norm, the last into the final norm (x in DRAM for the LM head)."""
        from models.demos.blackhole.qwen36.tt.fabric_boundary.ops import fabric_boundary

        fb = fabric_boundary(self.args, self.mesh_device)
        if getattr(self, "_fb_g_final", None) is None:  # norm weights as flat pages (first, eager, call)
            for layer in self.layers:
                layer._fb_g_attn = fb.gamma(layer.attention_norm.norm.weight)
                layer._fb_g_ffn = fb.gamma(layer.ffn_norm.norm.weight)
            self._fb_g_final = fb.gamma(self.norm.norm.weight)
        rows = int(x.shape[-2])
        res, h = fb(x, None, self.layers[0]._fb_g_attn)
        ttnn.deallocate(x)
        n = len(self.layers)
        # QWEN35_FB_CONSUMER=1: each layer's last boundary runs the next layer's in-projection in its program (the
        # projection's weights stream while the boundary computes; decode_linear/ops.py)
        consumer = os.environ.get("QWEN35_FB_CONSUMER", "0") == "1" and rows == 1
        proj = None
        for i, layer in enumerate(self.layers):
            last = i == n - 1
            g_next = self._fb_g_final if last else self.layers[i + 1]._fb_g_attn
            nxt = self.layers[i + 1].attention.inproj_consumer(fb) if consumer and not last else None
            res, h, proj = layer.forward_decode_fabric(fb, res, h, g_next, cos, sin, cur_pos_tensor, page_table,
                                                       last=last, proj=proj, next_consumer=nxt)
        ttnn.deallocate(res)
        return h

    def _final_norm_decode_replicated(self, x):
        """The final RMSNorm on the replicated decode residual: the same sharded norm program and output layout as the
        TP path (lm_head norm config), without its all-gather."""
        from models.tt_transformers.tt.common import Mode

        nc = dict(self.args.get_norm_config("lm_head", Mode.DECODE))
        nc["output_mem_config"] = ttnn.DRAM_MEMORY_CONFIG
        x_in = ttnn.to_memory_config(x, nc["sharded_output_config"])
        ttnn.deallocate(x)
        out = self.norm.norm(x_in, mode=Mode.DECODE, in_sharded=True, out_sharded=True, norm_config=nc)
        ttnn.deallocate(x_in)
        return out

    # ---- adaptive MTP: a second persistent B=1 GDN state set (the live request's state during an MTP phase) --------
    def _ensure_gdn_mtp_scratch(self):
        if getattr(self, "_gdn_mtp_scratch", None) is not None:
            return
        scratch = []
        for layer in self.layers:
            if layer.is_full_attention:
                continue
            dn = layer.attention
            saved = (dn.B, dn.rec_state, dn.conv_states, dn.conv_carry, dn._zero_conv0, dn._stable_state)
            dn.B = 1
            dn.reset_state()
            scratch.append((dn, dn.rec_state, dn.conv_states, dn.conv_carry, dn._zero_conv0))
            dn.B, dn.rec_state, dn.conv_states, dn.conv_carry, dn._zero_conv0, dn._stable_state = saved
        self._gdn_mtp_scratch = scratch

    def _bind_gdn_mtp_scratch(self):
        """Bind the MTP scratch onto every GDN layer (like _bind_gdn_prefill_scratch, a different buffer set)."""
        self._ensure_gdn_mtp_scratch()
        prev = []
        for dn, rec, conv, carry, zero0 in self._gdn_mtp_scratch:
            prev.append((dn, dn.B, dn.rec_state, dn.conv_states, dn.conv_carry, dn._zero_conv0, dn._stable_state))
            dn.B = 1
            dn.rec_state = rec
            dn.conv_states = conv
            dn.conv_carry = carry
            dn._zero_conv0 = zero0
            dn._stable_state = True
        return prev

    def _unbind_gdn_mtp_scratch(self, prev):
        self._unbind_gdn_prefill_scratch(prev)

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
    # ---- prefill: masked fixed buckets on persistent device inputs (eager or traced) ------------------------------
    # The stock masked-bucket forward uploads its tokens, RoPE tables, page tables, chunk start and (per GDN layer) the
    # length masks from the host on every call. Here every per-segment input is a PERSISTENT device tensor staged with
    # copy_host_to_device_tensor, the KV fill covers the whole bucket (pad rows land in a reserved scratch block), and
    # the next-token row is selected by a staged one-hot: the bucket forward is one fixed program per bucket, so it
    # can be captured once and replayed for any real length / chunk start (capture_prefill_bucket_traces).
    _BUCKET_INPUTS_ATTR = "_bucket_inputs_store"

    def _bucket_inputs(self, bucket, max_blocks=None):
        """Persistent device inputs of one bucket (allocated on first use; call before any trace capture).
        ``max_blocks`` = width of the page-table rows this model is driven with (fixed after the first call)."""
        from models.autoports.ornith_ai_ornith_1_5_35b_a3b.tt.prefill_len import PrefillLen

        store = self.__dict__.setdefault(self._BUCKET_INPUTS_ATTR, {})
        bi = store.get(bucket)
        if bi is not None:
            return bi
        dev = self.device
        rep = ttnn.ReplicateTensorToMesh(dev)
        block_size = self._prefill_block_size()
        rd = self.args.rope_head_dim
        conv_k = next(l.attention.K for l in self.layers if not l.is_full_attention)
        if getattr(self, "_bucket_max_blocks", None) is None:
            assert max_blocks is not None, "first _bucket_inputs call must give the page-table row width"
            self._bucket_max_blocks = int(max_blocks)
        max_blocks = int(self._bucket_max_blocks)
        bi = {
            "tok": ttnn.from_torch(
                torch.zeros(1, bucket, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=dev,
                mesh_mapper=rep,
            ),
            # MTP (QWEN35_MTP): the head's prompt rows run inside the bucket program (tt/mtp_serving.py) - the one-hot that
            # picks the hidden of the prompt's last position for the first draft, and the previous segment's last hidden
            # (row 0 of the shifted hidden rows; zeroed by the host at a request's first segment)
            **(
                {
                    "mtp_sel": ttnn.from_torch(
                        torch.zeros(1, 1, 32, bucket, dtype=torch.bfloat16),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=dev,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=rep,
                    ),
                    "mtp_prev": ttnn.from_torch(
                        torch.zeros(1, 1, 1, self.args.dim, dtype=torch.bfloat16),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=dev,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=rep,
                    ),
                }
                if int(os.environ.get("QWEN35_MTP", "0") or 0) > 0
                else {}
            ),
            "cos": ttnn.from_torch(
                torch.zeros(1, 1, bucket, rd, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                mesh_mapper=rep,
            ),
            "sin": ttnn.from_torch(
                torch.zeros(1, 1, bucket, rd, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                mesh_mapper=rep,
            ),
            "full_pt": ttnn.from_torch(
                torch.zeros(1, max_blocks, dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=dev,
                mesh_mapper=rep,
            ),
            "chunk_pt": ttnn.from_torch(
                torch.zeros(1, bucket // block_size, dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=dev,
                mesh_mapper=rep,
            ),
            "csi": ttnn.from_torch(
                torch.zeros(1, dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=dev,
                mesh_mapper=rep,
            ),
            "len": PrefillLen(dev, bucket, conv_k),
            "staged": None,  # (tokens tuple, actual_len, chunk_start, page-table row tuple) last staged
        }
        store[bucket] = bi
        return bi

    def _prefill_block_size(self):
        from models.demos.blackhole.qwen36.tt.model import get_block_size

        return get_block_size(self._paged_kv_caches)

    def _stage_bucket_inputs(self, bi, token_ids, page_table, actual_len, chunk_start, bucket):
        """Host -> device copies of one segment's inputs (no allocation, no program)."""
        dev = self.device
        rep = ttnn.ReplicateTensorToMesh(dev)
        block_size = self._prefill_block_size()
        real = token_ids[:, :actual_len].to(torch.int32)
        tok = (
            torch.cat([real, torch.zeros(1, bucket - actual_len, dtype=torch.int32)], dim=1)
            if bucket > actual_len
            else real
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(tok, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=None, mesh_mapper=rep),
            bi["tok"],
        )
        cos_t, sin_t = self._rope_tp_cos_sin_torch(chunk_start, bucket)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None, mesh_mapper=rep),
            bi["cos"],
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None, mesh_mapper=rep),
            bi["sin"],
        )
        # page tables: the full row (fixed width) and the bucket's blocks; blocks past the real tokens point at the
        # reserved scratch block so the fixed-width KV fill of the pad rows never touches a live block
        pt = page_table.to(torch.int32).reshape(1, -1)
        max_blocks = int(bi["full_pt"].shape[-1])
        scratch = int(self._kv_scratch_block)
        if pt.shape[1] < max_blocks:
            pt = torch.cat([pt, torch.full((1, max_blocks - pt.shape[1]), scratch, dtype=torch.int32)], dim=1)
        elif pt.shape[1] > max_blocks:
            raise ValueError(
                f"page-table row has {pt.shape[1]} blocks but the bucket inputs were sized for {max_blocks} (first call); "
                "a wider block table would be truncated -> size the generator's page table for the largest request"
            )
        blk0 = chunk_start // block_size
        blkN = -(-(chunk_start + actual_len) // block_size)
        nblk = bucket // block_size
        cpt = torch.full((1, nblk), scratch, dtype=torch.int32)
        n_real = min(blkN - blk0, nblk, pt.shape[1] - blk0)
        cpt[:, :n_real] = pt[:, blk0 : blk0 + n_real]
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=None, mesh_mapper=rep),
            bi["full_pt"],
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(cpt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=None, mesh_mapper=rep),
            bi["chunk_pt"],
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.tensor([chunk_start], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=None,
                mesh_mapper=rep,
            ),
            bi["csi"],
        )
        bi["len"].stage(actual_len)

    def _run_bucket_dev(self, bi, bucket, chunk_start):
        """The fixed bucket program over the staged inputs: embedding -> layers -> one-hot row select -> norm ->
        LM head. Returns ``(logits, logits_sharded)``: the replicated next-token logits [1, 1, vocab] (host / readiness
        path) and the pre-gather vocab-sharded logits [1, 1, 32, vocab/TP] whose row 0 is the next token (rows 1..31 are
        the zero rows of the 32-row selector) — the exact spec of the decode sampler's input, so the serving adapter can
        sample the prefill token on device with the programs the decode warm-up compiled (stage 9)."""
        x = self._run_bucket_hidden(bi, bucket, chunk_start)
        vlen = bi["len"]
        mtp_hook = self.__dict__.get("_mtp_bucket_hook")
        if mtp_hook is not None:
            mtp_hook(bi, bucket, chunk_start, x)  # the MTP head's prompt rows over this segment (part of the trace)
        x_last = ttnn.matmul(vlen.logit_sel, x)  # [1,1,32,dim]: row 0 = hidden row actual_len-1, rows 1..31 zero
        ttnn.deallocate(x)
        x_last = ttnn.to_memory_config(x_last, ttnn.DRAM_MEMORY_CONFIG)
        x_last = self.norm(x_last, mode=Mode.PREFILL)
        logits_sh = ttnn.linear(x_last, self.lm_head_weight, compute_kernel_config=self._lm_head_ckc())  # [1,1,32,V/TP]
        if self._lmhead_vocab_sharded:
            from models.tt_transformers.tt.ccl import tt_all_gather

            # the all-gather frees its input: gather a copy so the sharded logits stay alive for the on-device sampler
            logits = tt_all_gather(
                ttnn.clone(logits_sh, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                self.mesh_device,
                self.tt_ccl,
                cluster_axis=None,
                dim=3,
                topology=self.args.ccl_topology(),
            )
        else:
            logits = ttnn.clone(logits_sh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        logits = ttnn.slice(logits, (0, 0, 0, 0), (1, 1, 1, logits.shape[-1]))  # row 0 -> [1,1,1,vocab]
        return ttnn.reshape(logits, (1, 1, logits.shape[-1])), logits_sh

    def _run_bucket_hidden(self, bi, bucket, chunk_start):
        """Embedding -> all layers over the staged bucket inputs; returns hidden [1, 1, bucket, dim] (device)."""
        x = self.embd(bi["tok"])
        x = ttnn.reshape(x, (1, 1, bucket, x.shape[-1]))
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x = self._apply_vision_merge(x, length=bucket)
        vlen = bi["len"]
        for layer in self.layers:
            if layer.is_full_attention:
                x_new = layer.forward(
                    x,
                    cos=bi["cos"],
                    sin=bi["sin"],
                    mode="prefill",
                    page_table=bi["full_pt"],
                    chunk_page_table=bi["chunk_pt"],
                    chunk_start_idx=chunk_start,  # host int: sizes nothing here (fixed-width tables), flex SDPA uses csi
                    chunk_start_idx_tensor=bi["csi"],
                    valid_len=vlen,
                )
            else:
                x_new = layer.forward(x, mode="prefill", chunk_size=self.args.gdn_chunk_size, valid_len=vlen)
            ttnn.deallocate(x)
            x = x_new
        return x

    def prefill_masked_bucket(
        self,
        token_ids,
        page_table,
        actual_len,
        chunk_start=0,
        bucket=None,
        flex_sdpa=True,
        vision_tokens=None,
        vis_row_offset=0,
        return_sharded=False,
        rope_staged=False,
    ):
        """Masked fixed-bucket prefill on persistent device inputs; replays the bucket's trace when one was captured
        (:meth:`capture_prefill_bucket_traces`). Falls back to the stock path for multimodal segments or when no KV
        scratch block is reserved (``_kv_scratch_block``). Returns the replicated next-token logits [1, 1, vocab];
        with ``return_sharded`` the pair ``(logits, logits_sharded [1,1,32,vocab/TP])`` (see :meth:`_run_bucket_dev`).
        """
        # multimodal segments run on the persistent-input path too once the trace-safe splice buffers exist
        # (`_alloc_vision_merge_buffers`, before the captures): the bucket body's where(mask, vision, text) selects the
        # staged image rows; without the buffers (text-only build) they fall back to the stock path
        vision_needs_stock = vision_tokens is not None and getattr(self, "_vis_buf", None) is None
        if vision_needs_stock or getattr(self, "_kv_scratch_block", None) is None or not flex_sdpa:
            if return_sharded:
                raise NotImplementedError(
                    "sharded prefill logits need the persistent-input bucket path (KV scratch block)"
                )
            return super().prefill_masked_bucket(
                token_ids,
                page_table,
                actual_len,
                chunk_start=chunk_start,
                bucket=bucket,
                flex_sdpa=flex_sdpa,
                vision_tokens=vision_tokens,
                vis_row_offset=vis_row_offset,
            )
        assert token_ids.shape[0] == 1, "masked-bucket prefill is single-sequence"
        if bucket is None:
            bucket = self._mask_bucket_for(actual_len)
        assert 1 <= actual_len <= bucket, f"actual_len {actual_len} not in [1, {bucket}]"
        pt = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        if chunk_start == 0:
            self._reset_gdn_state_for_new_sequence()
            if not rope_staged:  # a multimodal request staged its M-RoPE table over the WHOLE prompt (generator)
                self._build_request_rope(token_ids[:, :actual_len], None)
        bi = self._bucket_inputs(bucket, max_blocks=pt.shape[-1])
        self._stage_bucket_inputs(bi, token_ids, pt, actual_len, chunk_start, bucket)
        mtp_stage = self.__dict__.get("_mtp_stage_hook")
        if mtp_stage is not None:
            mtp_stage(bi, bucket, chunk_start, actual_len)  # host writes of the MTP head's segment inputs
        if getattr(self, "_vis_buf", None) is not None:
            # image rows of this segment into the splice buffers (vision_tokens None clears the mask: text segment)
            self._set_vision_merge(token_ids[:, :actual_len], vision_tokens, vis_row_offset)
        tr = self.__dict__.get("_bucket_traces", {}).get(bucket)
        if tr is not None:
            ttnn.execute_trace(self.device, tr[0], cq_id=0, blocking=False)
            return (tr[1], tr[2]) if return_sharded else tr[1]
        out, out_sh = self._run_bucket_dev(bi, bucket, chunk_start)
        ttnn.synchronize_device(self.device)
        return (out, out_sh) if return_sharded else out

    def capture_prefill_bucket_traces(self, page_table_row, buckets=None):
        """Capture one trace per masked bucket (after every prefill program is warm and every persistent allocation is
        made). A replay then serves any (actual_len, chunk_start) through the staged inputs."""
        buckets = list(buckets or self._PREFILL_MASK_BUCKETS)
        traces = self.__dict__.setdefault("_bucket_traces", {})
        pt = page_table_row if isinstance(page_table_row, torch.Tensor) else ttnn.to_torch(page_table_row)
        for b in sorted(buckets):
            if b in traces:
                continue
            bi = self._bucket_inputs(b, max_blocks=pt.shape[-1])
            toks = torch.zeros(1, b, dtype=torch.int32)
            self._reset_gdn_state_for_new_sequence()
            self._build_request_rope(toks[:, : max(1, b // 2)], None)
            self._stage_bucket_inputs(bi, toks, pt, max(1, b // 2), 0, b)
            ttnn.synchronize_device(self.device)
            tid = ttnn.begin_trace_capture(self.device, cq_id=0)
            out, out_sh = self._run_bucket_dev(bi, b, 0)
            ttnn.end_trace_capture(self.device, tid, cq_id=0)
            traces[b] = (tid, out, out_sh)
            self._reset_gdn_state_for_new_sequence()
        return traces

    def release_prefill_bucket_traces(self):
        for tid, *_ in self.__dict__.get("_bucket_traces", {}).values():
            ttnn.release_trace(self.device, tid)
        self._bucket_traces = {}

    def prefill_all_logits(self, token_ids, page_table, valid_len=None):
        """B=1 prefill returning host logits ``[1, T, vocab]`` for EVERY position (fills KV + GDN state as the regular
        prefill does). Runs on the WARMED persistent-input bucket programs (2048-token segments, masked tail bucket):
        an exact-length pass would compile new programs after the traces are parked, and a program compiled after a
        trace capture can have its kernel binaries overwritten by the next replay (tt-metal #48536; reproduced here as
        a mesh-wide NoC hang, `doc/optimized_full_model/triage/seq_stock_2048_traced_2048_stock_2049.txt`). Falls back
        to the exact-length path when no KV scratch block is reserved."""
        if getattr(self, "_kv_scratch_block", None) is None:
            return self._prefill_all_logits_exact(token_ids, page_table, valid_len)
        B, T = token_ids.shape
        assert B == 1
        vlen = int(valid_len or T)
        pt = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        outs = []
        seg = self.PREFILL_SEGMENT
        for cs in range(0, vlen, seg):
            n = min(seg, vlen - cs)
            bucket = self._mask_bucket_for(n)
            if cs == 0:
                self._reset_gdn_state_for_new_sequence()
                self._build_request_rope(token_ids[:, :vlen], None)
            bi = self._bucket_inputs(bucket, max_blocks=pt.shape[-1])
            self._stage_bucket_inputs(bi, token_ids[:, cs : cs + n], pt, n, cs, bucket)
            x = self._run_bucket_hidden(bi, bucket, cs)  # [1,1,bucket,dim]
            x = self.norm(x, mode=Mode.PREFILL)
            logits = self._lm_head(x)  # replicated [1,1,bucket,vocab]
            ttnn.deallocate(x)
            host = ttnn.to_torch(ttnn.get_device_tensors(logits)[0]).float()  # one replica (the others are identical)
            ttnn.deallocate(logits)
            outs.append(host.reshape(-1, bucket, self.args.vocab_size)[:1, :n])
        return torch.cat(outs, dim=1)  # [1, vlen, vocab]

    def warmup_all_logits_programs(self, page_table_row):
        """Compile the all-positions norm + LM-head programs of every bucket (before any trace capture)."""
        pt = page_table_row if isinstance(page_table_row, torch.Tensor) else ttnn.to_torch(page_table_row)
        for b in self._PREFILL_MASK_BUCKETS:
            # the full bucket (b tokens -> bucket b: the bucket-b norm / LM-head programs) AND a masked length that still
            # selects bucket b (b - 1); `max(1, b // 2)` alone selected bucket b // 2 and never compiled 2048 (stage-7 review)
            for n in (b, max(1, b - 1)):
                self.prefill_all_logits(torch.zeros(1, n, dtype=torch.int32), pt)
        ttnn.synchronize_device(self.device)

    def _prefill_all_logits_exact(self, token_ids, page_table, valid_len=None):
        """Stage-6 exact-length single pass (masked to ``valid_len``); compiles per length."""
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
        mc = mc.with_(
            **{
                k: (tuple(v) if isinstance(v, list) else v)
                for k, v in json.loads(os.environ["QWEN35_MC_OVERRIDE"]).items()
            }
        )
    # datatype-sweep precision policy (stage 8): the shipped doc/datatype_sweep/selected_precision_config.json, or the file
    # named by QWEN35_PRECISION_CONFIG (``none`` = code defaults), mapped onto the MultichipConfig knobs the runtime reads.
    policy, policy_path = precision.load_policy(num_devices=mesh_device.get_num_devices())
    if os.environ.get("QWEN35_GDN_FUSED_DECODE", "0") == "1":
        # the fused GDN decode steps the recurrent state in place in fp32 (kda.recurrent_decode)
        policy = {**policy, "linear_state_dtype": "fp32"}
    mc = precision.apply_policy(mc, policy)
    logger.info(
        f"precision policy {policy.get('config_id')} ({policy_path or 'code defaults'}): "
        f"{json.dumps(precision.describe_config(mc), sort_keys=True)}"
    )
    check_mesh_shape(mesh_device)
    if mc.mesh_shape is not None and tuple(mesh_device.shape) != tuple(mc.mesh_shape):
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
    model.precision_policy = policy
    # the policy read back from the built tensors / kernel configs; every mismatch is a construction bug
    eff = precision.effective_policy(model, args)
    bad = precision.check_effective(policy, eff)
    print(
        "PRECISION_EFFECTIVE "
        + json.dumps({"config_id": policy.get("config_id"), "mismatches": bad, **eff}, sort_keys=True)
    )
    if bad:
        raise RuntimeError(f"precision policy {policy.get('config_id')} not constructed as requested: {bad}")
    return args, model
