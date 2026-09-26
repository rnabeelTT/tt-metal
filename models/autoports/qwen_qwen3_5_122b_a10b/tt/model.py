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

import json
import os
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tt import precision
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
        from models.autoports.qwen_qwen3_5_122b_a10b.tt.prefill_len import PrefillLen

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
    ):
        """Masked fixed-bucket prefill on persistent device inputs; replays the bucket's trace when one was captured
        (:meth:`capture_prefill_bucket_traces`). Falls back to the stock path for multimodal segments or when no KV
        scratch block is reserved (``_kv_scratch_block``). Returns the replicated next-token logits [1, 1, vocab];
        with ``return_sharded`` the pair ``(logits, logits_sharded [1,1,32,vocab/TP])`` (see :meth:`_run_bucket_dev`).
        """
        if vision_tokens is not None or getattr(self, "_kv_scratch_block", None) is None or not flex_sdpa:
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
            self._build_request_rope(token_ids[:, :actual_len], None)
        bi = self._bucket_inputs(bucket, max_blocks=pt.shape[-1])
        self._stage_bucket_inputs(bi, token_ids, pt, actual_len, chunk_start, bucket)
        if getattr(self, "_vis_buf", None) is not None:
            self._set_vision_merge(token_ids[:, :actual_len], None, vis_row_offset)  # clears the mask
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
        seg = 2048
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
    policy, policy_path = precision.load_policy()
    mc = precision.apply_policy(mc, policy)
    logger.info(
        f"precision policy {policy.get('config_id')} ({policy_path or 'code defaults'}): "
        f"{json.dumps(precision.describe_config(mc), sort_keys=True)}"
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
