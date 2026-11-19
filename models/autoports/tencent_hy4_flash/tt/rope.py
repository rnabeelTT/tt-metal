# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4 RoPE (HF HYV4FlashRotaryEmbedding + apply_rotary_pos_emb) for text positions.

For text the three M-RoPE planes carry the same position, so the recomposition is the identity and this is plain
interleaved ("GLM-style") RoPE on the trailing `rope_dim` (64) channels: pair (2i, 2i+1) rotates by pos * inv_freq[i].
HF computes cos / sin in fp32, casts them to bf16 (the activations' dtype), then rotates in fp32 and casts back:

    y = bf16(x.f32 * cos_rep + rotate_half(x).f32 * sin_rep),  rotate_half(x)[2i] = -x[2i+1], [2i+1] = x[2i]

cos_rep = repeat_interleave(cos[:32], 2). On the device the rotate_half is an exact +-1 permutation matmul.
"""

from __future__ import annotations

import torch
import ttnn

from models.autoports.tencent_hy4_flash.tt.common import ckc, upload


def cos_sin(positions: torch.Tensor, theta: float, dim: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """-> bf16 cos_rep, sin_rep [n, dim] exactly as HF builds them for these positions."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    freqs = positions.reshape(-1, 1).float() * inv_freq.reshape(1, -1)  # [n, dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)
    cos = cos[..., : dim // 2].repeat_interleave(2, dim=-1)
    sin = sin[..., : dim // 2].repeat_interleave(2, dim=-1)
    return cos, sin


def rotate_half_matrix(dim: int = 64) -> torch.Tensor:
    """P with x @ P == rotate_half(x) (interleaved pairs)."""
    P = torch.zeros(dim, dim)
    for i in range(dim // 2):
        P[2 * i + 1, 2 * i] = -1.0  # out[2i] = -x[2i+1]
        P[2 * i, 2 * i + 1] = 1.0  # out[2i+1] = x[2i]
    return P


def rope_torch(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Reference rotation of x [..., n, dim] (bf16) with cos / sin [n, dim] bf16 (host use, tests)."""
    xf = x.float()
    rh = torch.stack((-xf[..., 1::2], xf[..., 0::2]), dim=-1).flatten(-2)
    return (xf * cos.float() + rh * sin.float()).to(x.dtype)


class DeviceRope:
    """Rotate device tensors [..., n, 64] (TILE) with per-row cos / sin uploaded by the caller ([1, 1, n, 64])."""

    def __init__(self, mesh, dim: int = 64):
        self.mesh, self.dim = mesh, dim
        self.P = upload(mesh, rotate_half_matrix(dim).reshape(1, 1, dim, dim), ttnn.float32)
        self.ckc = ckc(mesh, fp32=True)

    def __call__(self, x, cos, sin, mem=None):
        """x bf16 [b, h, n, 64]; cos / sin fp32 or bf16 [1, 1, n, 64] (broadcast over b, h). -> bf16."""
        mem = mem or ttnn.DRAM_MEMORY_CONFIG
        xf = ttnn.typecast(x, ttnn.float32, memory_config=mem)
        rh = ttnn.matmul(xf, self.P, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=mem)
        y = ttnn.add(ttnn.multiply(xf, cos, memory_config=mem), ttnn.multiply(rh, sin, memory_config=mem),
                     memory_config=mem)
        return ttnn.typecast(y, ttnn.bfloat16, memory_config=mem)

    def tables(self, positions: torch.Tensor, theta: float, shard_rows: bool = False):
        """Upload fp32 copies of the bf16 cos / sin for these positions: [1, 1, n_pad32, 64] each (shard_rows: split
        over the dies by rows, die d holding the d-th quarter; n a multiple of 32 x dies)."""
        n = int(positions.numel())
        npad = -(-n // 32) * 32
        pos = torch.zeros(npad, dtype=torch.long)
        pos[:n] = positions.reshape(-1)
        c, s = cos_sin(pos, theta, self.dim)
        mp = ("shard", 2) if shard_rows else "rep"
        return (
            upload(self.mesh, c.float().reshape(1, 1, npad, self.dim), ttnn.float32, mapper=mp),
            upload(self.mesh, s.float().reshape(1, 1, npad, self.dim), ttnn.float32, mapper=mp),
        )
