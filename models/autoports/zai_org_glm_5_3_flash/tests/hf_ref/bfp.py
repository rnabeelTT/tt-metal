# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pure-torch emulation of the tt-metal host block-float packing (no ttnn import: it also runs in the HF venv).

tt_metal/impl/data_format/blockfloat_common.cpp (convert_u32_to_bfp<Bfp*_b, false>, pack_fp32_vec_as_bfp_tiles):
one shared 8-bit exponent per 16 consecutive elements of a tile face row (the max fp32 exponent of the block); each
element keeps its sign and a W-bit magnitude (W = 3 for bfp4_b, 7 for bfp8_b, hidden bit included): the 24-bit
mantissa (hidden 1 | 23 bits) is shifted right by (shared - own exponent) (truncating), then rounded to W bits to
nearest, ties to even, and saturated at 2^W - 1. Zero / denormal inputs give 0. The value is
m * 2^(shared - 127 - (W - 1)).

A ttnn weight [K, N] in TILE layout groups 16 consecutive N (output) columns of one K row, so for an HF weight
W [out, in] (stored on device as W^T) the blocks run along `out` (dim -2). tools/bfp_check.py checks this module
bit-exactly against ttnn.from_torch -> to_torch on the checkpoint's expert weights.
"""

from __future__ import annotations

import torch

MANT_BITS = {"bfp4": 3, "bfp8": 7, "bfloat4_b": 3, "bfloat8_b": 7}


def bfp_round(w: torch.Tensor, fmt: str, dim: int = -2) -> torch.Tensor:
    """w rounded to the block-float format `fmt` (bfp4 / bfp8), blocks of 16 along `dim`; returned in w's dtype
    (bf16 holds every bfp4 / bfp8 value exactly)."""
    W = MANT_BITS[fmt]
    dt = w.dtype
    x = w.float().movedim(dim, -1).contiguous()
    n = x.shape[-1]
    assert n % 16 == 0, f"block dim {n} is not a multiple of 16"
    x = x.reshape(*x.shape[:-1], n // 16, 16)
    bits = x.view(torch.int32)
    exp = (bits >> 23) & 0xFF
    man = bits & 0x7FFFFF
    neg = bits < 0
    shared = exp.amax(dim=-1, keepdim=True)
    m = man | (1 << 23)
    diff = shared - exp
    m = torch.where(diff >= 24, torch.zeros_like(m), m >> diff.clamp(max=23))
    shift = 24 - W
    rv = m & ((1 << shift) - 1)
    m = m >> shift
    tie = 1 << (shift - 1)
    up = (rv > tie) | ((rv == tie) & ((m & 1) == 1))
    m = (m + up.to(m.dtype)).clamp(max=(1 << W) - 1)
    m = torch.where(exp == 0, torch.zeros_like(m), m)
    val = torch.ldexp(m.double(), (shared - 127 - (W - 1)).double())
    val = torch.where(neg, -val, val)
    out = val.reshape(*val.shape[:-2], n).movedim(-1, dim)
    return out.to(dt)


def quantize_state(sd: dict, spec: dict) -> dict:
    """Checkpoint-named layer state dict, in place, with the weights matching a pattern of `spec` ({fnmatch pattern:
    fmt}, first match wins) rounded as the port stores them ([out, in] weights, blocks along out)."""
    import fnmatch

    for k in list(sd):  # in place: each original is freed as it is replaced (192 experts per layer)
        for pat, fmt in spec.items():
            if fnmatch.fnmatchcase(k, pat):
                assert sd[k].dim() == 2, (k, tuple(sd[k].shape))
                sd[k] = bfp_round(sd[k], fmt, dim=0)
                break
    return sd
