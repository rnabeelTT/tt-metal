# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Inkling's relative position bias, in the form the TT attention kernels consume.

The model (transformers `InklingRelativeLogits`): for query position q and key position k, head h,

    bias[h, q, k] = (r_q[h] @ proj)[q - k]   if 0 <= q - k < E   else 0

with r_q[h] = r_proj(x_q).view(heads, d_rel)[h] (d_rel = 16), proj a learned [d_rel, E] table, E = rel_extent (1024 on
global layers, the 512 window on sliding layers). Global layers past 128K positions also multiply the bias by tau(q).

The kernels add the bias to Q K^T *before* the softmax scale (exp((QK + bias - max) * scale)), and score 32x32 tiles:
a tile row is one (row, head) query, a tile column one key position. For a query at pos = 32 a + phi and a key tile
starting at base = 32 b, column c is distance d = 32 m + phi - c with m = a - b. So a bias tile is a matmul

    bias_tile = Rq @ T[phi][m]      Rq: [32 rows, 32] (r_q in columns 0..15, zero-padded), T[phi][m]: [32, 32]

where T[phi][m][j, c] = proj[j, 32 m + phi - c] / scale (zero where the distance is outside [0, E)), precomputed per layer
for the 32 phases: [32 phases, 32, (M) * 32] with M = E / 32 + 1 tiles per phase. The reader DMAs whole pre-tilized
tiles; no per-element work on the data-movement cores. Rows at different positions (spec verify) each use their own
(phi, m): the compute kernel accumulates one small matmul per row, with that row's Rq_r (only its tile rows nonzero).

This module builds the tables and emulates the kernel's arithmetic on the host (the tests check it against HF).
"""

from __future__ import annotations

import torch

TILE = 32


def n_tiles(extent: int) -> int:
    """Tiles per phase: distances 32 m + phi - c for c in [0, 32) cover [0, E) for m in [0, E / 32]."""
    assert extent % TILE == 0, extent
    return extent // TILE + 1


def build_tables(proj: torch.Tensor, scale: float) -> torch.Tensor:
    """proj [d_rel, E] -> T [32 phases, 32, n_tiles(E) * 32] fp32 (rows >= d_rel zero), values proj / scale."""
    d_rel, E = proj.shape
    assert d_rel <= TILE
    M = n_tiles(E)
    phi = torch.arange(TILE).view(TILE, 1, 1)  # [32, 1, 1]
    m = torch.arange(M).view(1, M, 1)
    c = torch.arange(TILE).view(1, 1, TILE)
    d = (TILE * m + phi - c).reshape(TILE, M * TILE)  # [phases, M*32] distance per column
    valid = (d >= 0) & (d < E)
    T = torch.zeros(TILE, TILE, M * TILE, dtype=torch.float32)
    g = proj.float()[:, d.clamp(0, E - 1)]  # [d_rel, phases, M*32]
    T[:, :d_rel, :] = (g * valid).permute(1, 0, 2) / scale
    return T


def rq_tiles(r: torch.Tensor, G: int) -> torch.Tensor:
    """r [R, G, d_rel] (one kv head's G query heads, R rows) -> per-row Rq tiles [R, 32, 32]: tile row r*G+i holds
    r[r, i] in columns 0..d_rel-1 of tile R_r, every other row zero."""
    R, G_, d_rel = r.shape
    assert G_ == G and R * G <= TILE
    out = torch.zeros(R, TILE, TILE, dtype=torch.float32)
    for j in range(R):
        out[j, j * G : (j + 1) * G, :d_rel] = r[j].float()
    return out


def emulate_bias(r: torch.Tensor, pos: list[int], T: torch.Tensor, kv_len: int) -> torch.Tensor:
    """The kernel's bias over key positions [0, kv_len) for R rows at positions `pos` (r [R, G, d_rel]):
    -> [R*G, kv_len] (rows past the causal limit get the table's zeros: causality is the separate mask's job).
    Values are bias / scale (what is added to the raw scores)."""
    R, G, _ = r.shape
    Rq = rq_tiles(r, G)
    M = T.shape[-1] // TILE
    out = torch.zeros(TILE, (kv_len + TILE - 1) // TILE * TILE)
    for b in range(out.shape[1] // TILE):
        acc = torch.zeros(TILE, TILE)
        for j in range(R):
            a, phi = divmod(pos[j], TILE)
            m = a - b
            if 0 <= m < M:
                acc += Rq[j] @ T[phi, :, m * TILE : (m + 1) * TILE]
        out[:, b * TILE : (b + 1) * TILE] = acc
    return out[: R * G, :kv_len]


def reference_bias(r: torch.Tensor, pos: list[int], proj: torch.Tensor, kv_len: int) -> torch.Tensor:
    """HF's formula for the same rows: [R*G, kv_len] (bias, unscaled; zero outside 0 <= q-k < E incl. q < k)."""
    R, G, _ = r.shape
    E = proj.shape[1]
    rel = r.float() @ proj.float()  # [R, G, E]
    k = torch.arange(kv_len)
    out = torch.zeros(R, G, kv_len)
    for j in range(R):
        d = pos[j] - k
        ok = (d >= 0) & (d < E)
        out[j][:, ok] = rel[j][:, d[ok]]
    return out.reshape(R * G, kv_len)
