# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Static configuration for the Qwen3.5-9B Gated DeltaNet (linear attention) layer."""
from dataclasses import dataclass


@dataclass(frozen=True)
class GDNConfig:
    num_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel_size: int
    norm_eps: float
    q_dim: int
    k_dim: int
    v_dim: int
    long_prefill_chunk_size: int = 128
    # Projection weight dtype (qkv/z/a/b/out); default keeps the validated bfloat8_b.
    weight_dtype: object = None
    # Optional decode matmul hook (x, weight, role, ckc, mc) -> out|None for roles "mega"/"o" (args.gdn_decode_matmul_fn).
    decode_matmul_fn: object = None
    # Optional per-role weight memory configs {"mega": ..., "o": ...} (args.gdn_weight_memcfgs).
    weight_memcfgs: object = None
    # fp32 recurrent decode step (args.gdn_decode_high_precision); pair with an fp32 external state buffer.
    decode_high_precision: bool = False

    @classmethod
    def from_args(cls, args) -> "GDNConfig":
        return cls(
            num_heads=args.linear_num_key_heads,
            num_v_heads=args.linear_num_value_heads,
            head_k_dim=args.linear_key_head_dim,
            head_v_dim=args.linear_value_head_dim,
            conv_kernel_size=args.linear_conv_kernel_dim,
            norm_eps=args.norm_eps,
            q_dim=args.linear_q_dim,
            k_dim=args.linear_k_dim,
            v_dim=args.linear_v_dim,
            weight_dtype=getattr(args, "gdn_weight_dtype", None),
            decode_matmul_fn=getattr(args, "gdn_decode_matmul_fn", None),
            weight_memcfgs=getattr(args, "gdn_weight_memcfgs", None),
            decode_high_precision=bool(getattr(args, "gdn_decode_high_precision", False)),
        )
