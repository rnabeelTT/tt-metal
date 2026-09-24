# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Functional TTNN decoder layer for ``Qwen/Qwen3.5-122B-A10B`` (stage 1 of the autoport).

One ``FunctionalDecoder`` is one HF ``Qwen3_5MoeDecoderLayer``: a token mixer (Gated DeltaNet for
``linear_attention`` layers, gated full attention with a paged KV cache for ``full_attention`` layers),
zero-centered RMSNorms, and the 256-expert MoE feed-forward. The token mixers and norms are the
validated qwen36 (Qwen3.5/3.6) implementations; the MoE block is :mod:`.moe`.

Public tensor contract (single device, ``[B, T, H]`` activations like qwen36)
-----------------------------------------------------------------------------
``prefill_forward(hidden_states, *, start_pos=0, page_table=None, user_id=0, valid_len=None)``
    ``hidden_states``: ``[B, T, 3072]`` or ``[1, B, T, 3072]`` bf16 TILE tensor on device. Any logical
    ``T <= supported context`` is accepted: the layer pads internally to the GDN chunk (128) and masks
    the padding (``valid_len``), then slices the output back to ``T``. Full-attention layers need a
    paged KV cache (:meth:`allocate_kv_cache`) and ``page_table`` (device int32 ``[B, blocks]``); K/V
    for positions ``[start_pos, start_pos+T)`` are written through ``paged_fill_cache``. Linear-attention
    layers update their recurrent/conv state (:meth:`allocate_gdn_state`) in place.
``decode_forward(hidden_states, *, current_pos, page_table=None, cos=None, sin=None)``
    ``hidden_states``: ``[B, 1, 3072]`` or ``[1, 1, B, 3072]``. ``current_pos``: device int32 ``[B]``.
    Full attention: ``paged_update_cache`` + ``paged_scaled_dot_product_attention_decode``; ``cos``/``sin``
    are device tensors ``[B, 1, 64]`` (see :meth:`rope_decode_host`), so the whole step is trace-safe.
    Linear attention: recurrent update of the external state buffers in place.
Both return a tensor shaped like the input. No host/torch work happens inside a forward pass; weight
conversion lives in :meth:`from_state_dict` and state allocation in the ``allocate_*`` helpers.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_5_122b_a10b.tt.layer import Qwen35MoeDecoderLayer
from models.autoports.qwen_qwen3_5_122b_a10b.tt.model_config import Qwen35MoeModelArgs
from models.autoports.qwen_qwen3_5_122b_a10b.tt.weights import remap_hf_layer_state
from models.common.lightweightmodule import LightweightModule
from models.demos.blackhole.qwen36.tt.gdn.state import restore_split_conv_from_fused
from models.demos.blackhole.qwen36.tt.rope import Qwen36RoPESetup

GDN_CHUNK = 128  # the chunk-seq GDN prefill kernel's only supported chunk size
TILE = 32


def _text_config(hf_config):
    return hf_config.get_text_config() if hasattr(hf_config, "get_text_config") else hf_config


class FunctionalDecoder(LightweightModule):
    """Functional (correctness-first) TTNN implementation of one Qwen3.5-122B-A10B decoder layer."""

    LAYER_CLS = Qwen35MoeDecoderLayer  # subclasses (fused/optimized decoders) substitute their layer class

    def __init__(self, *, args, layer, layer_idx, mesh_device, rope):
        super().__init__()
        self.args = args
        self.layer = layer
        self.layer_idx = layer_idx
        self.mesh_device = mesh_device
        self.rope = rope
        self.layer_type = "full_attention" if layer.is_full_attention else "linear_attention"
        self.hidden_size = args.dim
        self.kv_cache = None
        self.gdn_state = None

    # ---------------------------------------------------------------------------------------------
    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        max_batch_size: int = 1,
        max_seq_len: int | None = None,
        tensor_cache_path: str | Path | None = None,
        moe_precision=None,
        _args_hook=None,
        **kwargs,
    ):
        """Build the layer from HF-format weights of layer ``layer_idx``.

        ``state_dict`` holds that layer's tensors under layer-local keys (``self_attn.q_proj.weight``,
        ``mlp.experts.gate_up_proj``, ...) or full-model keys (``model.language_model.layers.N.*``).
        ``hf_config`` is the top-level ``Qwen3_5MoeConfig`` or its text config; it must agree with the
        checkpoint selected by ``HF_MODEL`` (default ``Qwen/Qwen3.5-122B-A10B``).
        """
        text = _text_config(hf_config)
        if max_seq_len is None:
            max_seq_len = int(text.max_position_embeddings)
        args = Qwen35MoeModelArgs(mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len)
        if moe_precision is not None:
            args.moe_precision = moe_precision
        if _args_hook is not None:
            _args_hook(args)
        if int(text.hidden_size) != args.dim or int(text.num_hidden_layers) != args.n_layers:
            raise ValueError(
                f"hf_config (hidden {text.hidden_size}, layers {text.num_hidden_layers}) does not match the "
                f"HF_MODEL checkpoint (hidden {args.dim}, layers {args.n_layers})"
            )
        if text.layer_types[layer_idx] not in ("linear_attention", "full_attention"):
            raise ValueError(f"unsupported layer_type {text.layer_types[layer_idx]!r}")
        internal = remap_hf_layer_state(
            state_dict, layer_idx, linear_q_dim=args.linear_q_dim, linear_k_dim=args.linear_k_dim
        )
        cache = Path(tensor_cache_path) if tensor_cache_path else None
        layer = cls.LAYER_CLS(mesh_device, args, internal, layer_idx, cache, tt_ccl=None)
        rope = Qwen36RoPESetup(mesh_device, args) if layer.is_full_attention else None
        return cls(args=args, layer=layer, layer_idx=layer_idx, mesh_device=mesh_device, rope=rope)

    # ---------------------------------------------------------------------------------------------
    # State allocation (setup-time; explicit buffers so decode can be traced)
    def allocate_kv_cache(self, *, max_batch_size: int, max_seq_len: int, block_size: int = 64, dtype=ttnn.bfloat16):
        """Paged KV cache ``[num_blocks, n_kv_heads, block_size, head_dim]`` for this full-attention layer."""
        assert self.layer.is_full_attention, "only full_attention layers hold a KV cache"
        blocks_per_seq = math.ceil(max_seq_len / block_size)
        num_blocks = max_batch_size * blocks_per_seq
        shape = (num_blocks, self.args.n_kv_heads, block_size, self.args.head_dim)
        k = ttnn.zeros(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
        v = ttnn.zeros(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
        self.layer.attention.set_paged_kv_cache(k, v)
        self.kv_cache = {
            "k": k,
            "v": v,
            "block_size": block_size,
            "num_blocks": num_blocks,
            "blocks_per_seq": blocks_per_seq,
        }
        return self.kv_cache

    def allocate_gdn_state(self, *, batch_size: int):
        """Zeroed recurrent ``[B, Nv, Dk, Dv]`` and conv ``[B, K-1, q+k+v]`` buffers, bound to the layer in place."""
        assert not self.layer.is_full_attention, "only linear_attention layers hold GDN state"
        dn = self.layer.attention
        state_dtype = getattr(self.args, "gdn_state_dtype", None) or ttnn.bfloat16
        rec = ttnn.zeros(
            (batch_size, dn.num_v_heads, dn.head_k_dim, dn.head_v_dim),
            dtype=state_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
        )
        D = dn.cfg.q_dim + dn.cfg.k_dim + dn.cfg.v_dim
        conv = ttnn.zeros(
            (batch_size, dn.conv_kernel_size - 1, D),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
        )
        # Decode conv shift-register: one [B,1,D] buffer per tap (address-stable; refreshed from the fused
        # [B,K-1,D] buffer after every prefill). Avoids the per-step row slices of the fused buffer.
        split = [
            ttnn.zeros((batch_size, 1, D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
            for _ in range(dn.conv_kernel_size - 1)
        ]
        dn.reset_state()
        dn.set_external_state(rec, conv)
        dn.split_conv_state = split
        self.gdn_state = {"recurrent": rec, "conv": conv, "split": split, "batch_size": batch_size}
        return self.gdn_state

    def reset_gdn_state(self):
        """Zero the external GDN buffers in place (new sequence) without changing their addresses."""
        st = self.gdn_state
        assert st is not None
        self._zero_inplace(st["recurrent"])
        self._zero_inplace(st["conv"])
        for t in st["split"]:
            self._zero_inplace(t)
        dn = self.layer.attention
        for t in (dn.conv_state_q, dn.conv_state_k, dn.conv_state_v):
            if t is not None:
                ttnn.deallocate(t)
        dn.recurrent_state, dn.fused_conv_state = st["recurrent"], st["conv"]
        dn.conv_state_q = dn.conv_state_k = dn.conv_state_v = None
        dn.split_conv_state = st["split"]
        dn.use_inplace_state = True

    # ---------------------------------------------------------------------------------------------
    def rope_decode_host(self, position_ids: torch.Tensor):
        """Host bf16 cos/sin ``[B, 1, rope_dim]`` for decode positions (copy into persistent device buffers)."""
        assert self.rope is not None
        pos = position_ids.reshape(-1).to(torch.long)
        cos = self.rope.cos_cpu[pos].reshape(-1, 1, self.args.rope_head_dim)
        sin = self.rope.sin_cpu[pos].reshape(-1, 1, self.args.rope_head_dim)
        return cos, sin

    @staticmethod
    def _as_bth(x):
        """Accept ``[B,T,H]`` or ``[1,B,T,H]`` and return (``[B,T,H]`` tensor, original shape)."""
        orig = list(x.shape)
        if len(orig) == 4:
            assert orig[0] == 1, f"leading dim must be 1, got {orig}"
            return ttnn.reshape(x, orig[1:]), orig
        return x, orig

    # ---------------------------------------------------------------------------------------------
    def prefill_forward(self, hidden_states, *, start_pos: int = 0, page_table=None, user_id: int = 0, valid_len=None):
        x, orig = self._as_bth(hidden_states)
        B, T, H = x.shape
        assert B == 1, "prefill runs one user at a time (vLLM semantics); loop over users for B > 1"
        T_logical = T if valid_len is None else int(valid_len)
        T_pad = -(-T // GDN_CHUNK) * GDN_CHUNK
        if T_pad != T:
            x = ttnn.pad(x, [(0, 0), (0, T_pad - T), (0, 0)], value=0.0)
        # Chunk-continuation contract: start_pos is a multiple of the GDN chunk (128) and of the KV block size;
        # the chunked SDPA derives its program config from the lowest set bit of chunk_start_idx.
        assert start_pos % GDN_CHUNK == 0, f"start_pos {start_pos} must be a multiple of {GDN_CHUNK}"
        if self.layer.is_full_attention:
            assert page_table is not None and self.kv_cache is not None, "full_attention prefill needs a paged KV cache"
            bs = self.kv_cache["block_size"]
            assert start_pos % bs == 0, f"start_pos {start_pos} must be a multiple of the KV block size {bs}"
            # cos/sin for positions [start_pos, start_pos+T_pad) sliced from the resident device tables (no host work).
            cos = ttnn.slice(self.rope.cos_device, [0, start_pos, 0], [1, start_pos + T_pad, self.args.rope_head_dim])
            sin = ttnn.slice(self.rope.sin_device, [0, start_pos, 0], [1, start_pos + T_pad, self.args.rope_head_dim])
            # K/V are written for the logical tokens' blocks only (pad rows inside the last block are harmless and
            # are overwritten by decode); never beyond the user's page-table row.
            n_blocks = -(-(start_pos + T) // bs)
            assert (
                n_blocks <= page_table.shape[1]
            ), f"{start_pos + T} tokens need {n_blocks} blocks > page table width {page_table.shape[1]}"
            pt_row = (
                page_table
                if page_table.shape[0] == 1
                else ttnn.slice(page_table, [user_id, 0], [user_id + 1, page_table.shape[1]])
            )
            assert (
                valid_len is None
            ), "full_attention prefill takes the logical length from the tensor (no valid_len masking)"
            # paged_fill_cache writes input tile i into chunk_page_table[i // block_size]: the chunk's page table must
            # START at this chunk's first block (start_pos // bs), not at block 0.
            chunk_pt = ttnn.slice(pt_row, [0, start_pos // bs], [1, n_blocks])
            out = self.layer.forward(
                x,
                cos=cos,
                sin=sin,
                mode="prefill",
                page_table=pt_row,
                chunk_page_table=chunk_pt,
                chunk_start_idx=start_pos,
            )
            ttnn.deallocate(cos)
            ttnn.deallocate(sin)
            ttnn.deallocate(chunk_pt)
            if pt_row is not page_table:
                ttnn.deallocate(pt_row)
        else:
            out = self._gdn_prefill(x, T_logical if T_pad != T_logical else None, user_id, start_pos)
        if T_pad != T:
            ttnn.deallocate(x)
            out_full = out
            out = ttnn.slice(out_full, [0, 0, 0], [1, T, H])
            ttnn.deallocate(out_full)
        if len(orig) == 4:
            out = ttnn.reshape(out, orig)
        return out

    def _gdn_prefill(self, x, valid_len, user_id, start_pos):
        """Linear-attention prefill of ONE user; the resulting state lands in the persistent buffers.

        The qwen36 chunk-prefill path REASSIGNS the layer's recurrent/conv state tensors (it does not
        write the external buffers in place), so after the forward the new state is copied into row
        ``user_id`` of the batched external buffers and the layer is re-pointed at them. Decode then
        updates those buffers in place (trace-safe). ``start_pos > 0`` (chunk continuation) is
        supported for a single-user state, continuing from the layer's current conv/recurrent state.
        """
        dn = self.layer.attention
        st = self.gdn_state
        assert st is not None, "call allocate_gdn_state(batch_size=...) before prefill"
        B = st["batch_size"]
        rec_b, conv_b = st["recurrent"], st["conv"]
        assert 0 <= user_id < B
        if start_pos == 0:
            # Fresh sequence: start from zeroed DEVICE state buffers (no host init). B==1 zeroes the persistent
            # buffers in place; B>1 prefills into a zeroed B=1 scratch and writes row `user_id` below.
            if B == 1:
                self._zero_inplace(rec_b)
                self._zero_inplace(conv_b)
                rec_in, conv_in = rec_b, conv_b
            else:
                rec_in = ttnn.zeros(
                    (1,) + tuple(rec_b.shape)[1:], dtype=rec_b.dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device
                )
                conv_in = ttnn.zeros(
                    (1,) + tuple(conv_b.shape)[1:], dtype=conv_b.dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device
                )
            dn.recurrent_state, dn.fused_conv_state = rec_in, conv_in
            dn.conv_state_q = dn.conv_state_k = dn.conv_state_v = None
            dn.split_conv_state = None
            dn.use_inplace_state = True
        else:
            assert B == 1, "chunked prefill continuation is supported for single-user GDN state"
            assert dn.recurrent_state is not None and (
                dn.fused_conv_state is not None or dn.conv_state_q is not None
            ), "no state to continue from"
            if getattr(dn, "split_conv_state", None) is not None and dn.fused_conv_state is not None:
                # Decode advances only the per-tap shift-register buffers; rebuild the fused [B,K-1,D] window the
                # chunk kernel reads from them (a no-op right after a prefill, where both are in sync).
                window = ttnn.concat(list(dn.split_conv_state), dim=1)
                if window.layout != dn.fused_conv_state.layout:
                    window = ttnn.to_layout(window, dn.fused_conv_state.layout)
                ttnn.copy(window, dn.fused_conv_state)
                ttnn.deallocate(window)
        out = self.layer.forward(x, mode="prefill", chunk_size=GDN_CHUNK, valid_len=valid_len)
        # Materialize this user's post-prefill state as [1, ...] tensors.
        rec1 = dn.recurrent_state
        if dn.fused_conv_state is not None:
            conv1 = dn.fused_conv_state
        else:
            conv1 = ttnn.concat([dn.conv_state_q, dn.conv_state_k, dn.conv_state_v], dim=2)
            if conv1.layout != ttnn.TILE_LAYOUT:
                conv1 = ttnn.to_layout(conv1, ttnn.TILE_LAYOUT)
        for src, dst in ((rec1, rec_b), (conv1, conv_b)):
            if src.layout != dst.layout:
                src = ttnn.to_layout(src, dst.layout)
            if list(src.shape) != list(dst.shape):
                src = ttnn.reshape(src, (1,) + tuple(dst.shape)[1:]) if src.shape[0] == 1 and B == 1 else src
            if B == 1:
                ttnn.copy(src, dst)
                continue
            parts = []
            if user_id > 0:
                parts.append(ttnn.slice(dst, [0] + [0] * (len(dst.shape) - 1), [user_id] + list(dst.shape)[1:]))
            parts.append(src)
            if user_id + 1 < B:
                parts.append(ttnn.slice(dst, [user_id + 1] + [0] * (len(dst.shape) - 1), list(dst.shape)))
            merged = ttnn.concat(parts, dim=0)
            ttnn.copy(merged, dst)
            ttnn.deallocate(merged)
            for t in parts:
                if t is not src:
                    ttnn.deallocate(t)
        # Re-point the layer at the persistent buffers: decode updates the recurrent state and the per-tap conv
        # buffers in place; a chunk continuation (start_pos > 0) rebuilds the fused conv window from the taps.
        stale = [t for t in (rec1, conv1, dn.conv_state_q, dn.conv_state_k, dn.conv_state_v) if t is not None]
        if start_pos == 0 and B > 1:
            stale += [rec_in, conv_in]
        dn.recurrent_state, dn.fused_conv_state = rec_b, conv_b
        dn.conv_state_q = dn.conv_state_k = dn.conv_state_v = None
        # Refresh the decode shift-register buffers from the fused conv window (address-stable copies).
        dn.split_conv_state = st["split"]
        restore_split_conv_from_fused(dn)
        dn.use_inplace_state = True
        seen = set()
        for t in stale:
            if t is rec_b or t is conv_b or id(t) in seen:
                continue
            seen.add(id(t))
            ttnn.deallocate(t)
        return out

    @staticmethod
    def _zero_inplace(buf):
        zeros = ttnn.zeros_like(buf)
        ttnn.copy(zeros, buf)
        ttnn.deallocate(zeros)

    def decode_forward(self, hidden_states, *, current_pos, page_table=None, cos=None, sin=None):
        orig = list(hidden_states.shape)
        if len(orig) == 4:  # [1, 1, B, H] -> [B, 1, H]
            x = ttnn.reshape(hidden_states, (orig[2], 1, orig[3]))
        else:
            x = hidden_states
        if self.layer.is_full_attention:
            assert page_table is not None and cos is not None and sin is not None
            out = self.layer.forward(x, cos, sin, mode="decode", position_tensor=current_pos, page_table=page_table)
        else:
            out = self.layer.forward(x, mode="decode")
        if len(orig) == 4:
            out = ttnn.reshape(out, orig)
        return out

    def forward(self, hidden_states, *, mode: str, **kwargs):
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")
