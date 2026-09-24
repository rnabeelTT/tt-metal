# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionConfig:
    num_heads: int
    num_kv_heads: int
    head_dim: int
    norm_eps: float
    max_seq_len: int
    # One fused [q+gate | k | v] projection (shared-input matmul) instead of three; opt-in via args.attn_fuse_qkv.
    fuse_qkv: bool = False
    # Projection weight dtype (q/k/v/o and the fused qkv); default keeps the validated bfloat8_b.
    weight_dtype: object = None
    # Decode SDPA chunk sizes (None -> kernel default 32/64).
    sdpa_decode_q_chunk: int | None = None
    sdpa_decode_k_chunk: int | None = None
    # Optional decode matmul hook (x, weight, role, ckc, mc) -> out|None for roles "qkv"/"q"/"o" (args.attn_decode_matmul_fn).
    decode_matmul_fn: object = None
    # Optional per-role weight memory configs {"qkv": ..., "o": ...} (e.g. DRAM width-sharded) (args.attn_weight_memcfgs).
    weight_memcfgs: object = None

    @classmethod
    def from_args(cls, args) -> "AttentionConfig":
        return cls(
            num_heads=args.n_heads,
            num_kv_heads=args.n_kv_heads,
            head_dim=args.head_dim,
            norm_eps=args.norm_eps,
            max_seq_len=args.max_seq_len,
            fuse_qkv=bool(getattr(args, "attn_fuse_qkv", False)),
            weight_dtype=getattr(args, "attn_weight_dtype", None),
            sdpa_decode_q_chunk=getattr(args, "sdpa_decode_q_chunk", None),
            sdpa_decode_k_chunk=getattr(args, "sdpa_decode_k_chunk", None),
            decode_matmul_fn=getattr(args, "attn_decode_matmul_fn", None),
            weight_memcfgs=getattr(args, "attn_weight_memcfgs", None),
        )
