# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import ttnn


@dataclass(frozen=True)
class AttentionWeights:
    q_proj: ttnn.Tensor
    k_proj: ttnn.Tensor
    v_proj: ttnn.Tensor
    o_proj: ttnn.Tensor
    q_norm: ttnn.Tensor  # +1 pre-offset (zero-centered RMSNorm)
    k_norm: ttnn.Tensor
    qkv_proj: ttnn.Tensor = None  # fused [in, 2*Hq*D + 2*Hkv*D] = [q+gate | k | v]; None -> separate projections


def load_attention_weights(mesh_device, state_dict, tensor_cache_path=None, fuse_qkv=False) -> AttentionWeights:
    def load_2d(name):
        t = state_dict[f"{name}.weight"].T.contiguous()
        return ttnn.as_tensor(
            t,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(tensor_cache_path / f"self_attn.{name}.weight") if tensor_cache_path else None,
        )

    def load_norm(name):
        t = state_dict[f"{name}.weight"] + 1.0
        return ttnn.as_tensor(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(tensor_cache_path / f"self_attn.{name}.weight_offset") if tensor_cache_path else None,
        )

    qkv_proj = None
    if fuse_qkv:
        import torch

        fused = torch.cat(
            [state_dict["q_proj.weight"], state_dict["k_proj.weight"], state_dict["v_proj.weight"]], dim=0
        ).T.contiguous()  # [in, q+gate | k | v]
        qkv_proj = ttnn.as_tensor(
            fused,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(tensor_cache_path / "self_attn.qkv_proj.weight") if tensor_cache_path else None,
        )
        q = k = v = None
    else:
        q, k, v = load_2d("q_proj"), load_2d("k_proj"), load_2d("v_proj")
    return AttentionWeights(
        q_proj=q,
        k_proj=k,
        v_proj=v,
        o_proj=load_2d("o_proj"),
        q_norm=load_norm("q_norm"),
        k_norm=load_norm("k_norm"),
        qkv_proj=qkv_proj,
    )
