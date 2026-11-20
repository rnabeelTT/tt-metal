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


MROPE_SECTION = (12, 10, 10)  # config rope_parameters mrope_section (t, h, w), interleaved

# Image prompts (tt/vision_host.py): a sequence's rope positions are not its slot indices. POSMAPS[slot] maps slot
# indices to the 3 M-RoPE planes; rpos() applies the active slot's map (Hy4Model.set_slot), identity without one.
POSMAPS: dict = {}
_ACTIVE = [0]


def set_active(slot: int):
    _ACTIVE[0] = int(slot)


def rpos(slots: torch.Tensor, slot: int | None = None) -> torch.Tensor:
    m = POSMAPS.get(_ACTIVE[0] if slot is None else slot)
    return slots if m is None else m(slots)


def rkey(slot: int | None = None):
    """Cache-key part of the active map (None: plain positions)."""
    m = POSMAPS.get(_ACTIVE[0] if slot is None else slot)
    return None if m is None else id(m)


def rpos_rows(pos: torch.Tensor, slots) -> torch.Tensor:
    """Per-row maps (batched decode rows of several sequences): pos [n] slot indices, slots [n] their model slots."""
    slots = [int(s) for s in slots]
    if not any(s in POSMAPS for s in set(slots)):
        return pos
    out = pos.reshape(1, -1).repeat(3, 1)
    sl = torch.tensor(slots)
    for s in set(slots):
        if s in POSMAPS:
            m = sl == s
            out[:, m] = POSMAPS[s](pos[m])
    return out


def cos_sin(positions: torch.Tensor, theta: float, dim: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """-> bf16 cos_rep, sin_rep [n, dim] exactly as HF builds them for these positions ([n], or [3, n] M-RoPE planes:
    HF recomposition_frequencies, frequency i from plane 1 / 2 when i % 3 == 1 / 2 and i < 3 x section)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    if positions.dim() == 2 and positions.shape[0] == 3:
        f3 = positions.float()[..., None] * inv_freq.reshape(1, 1, -1)  # [3, n, dim/2]
        freqs = f3[0].clone()
        for j in (1, 2):
            idx = slice(j, MROPE_SECTION[j] * 3, 3)
            freqs[:, idx] = f3[j][:, idx]
    else:
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
        if positions.dim() == 2 and positions.shape[0] == 3:  # M-RoPE planes
            n = int(positions.shape[1])
            npad = -(-n // 32) * 32
            pos = torch.zeros(3, npad, dtype=torch.long)
            pos[:, :n] = positions
        else:
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
