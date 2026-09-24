# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
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

    @classmethod
    def from_args(cls, args) -> "AttentionConfig":
        return cls(
            num_heads=args.n_heads,
            num_kv_heads=args.n_kv_heads,
            head_dim=args.head_dim,
            norm_eps=args.norm_eps,
            max_seq_len=args.max_seq_len,
            fuse_qkv=bool(getattr(args, "attn_fuse_qkv", False)),
        )
