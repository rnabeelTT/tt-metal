# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.scatter_rows: in-place row scatter into a ROW_MAJOR cache (sentinel 0xFFFFFFFF skips)."""

import pytest
import torch

import ttnn

SENTINEL = 0xFFFFFFFF


@pytest.mark.parametrize("rows,width,cache_rows", [(3, 512, 256), (37, 512, 4096), (2048, 512, 4096), (32, 128, 64)])
def test_scatter_rows(device, rows, width, cache_rows):
    torch.manual_seed(0)
    cache = torch.randn(1, 1, cache_rows, width).bfloat16()
    src = torch.randn(rows, width).bfloat16()
    idx = torch.randperm(cache_rows)[:rows].to(torch.int64)
    idx[::5] = SENTINEL  # every 5th row skipped
    t_cache = ttnn.from_torch(cache, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    t_src = ttnn.from_torch(src, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    t_idx = ttnn.from_torch(idx.reshape(1, rows), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out = ttnn.scatter_rows(t_cache, t_src, t_idx)
    assert out.buffer_address() == t_cache.buffer_address()
    ref = cache.clone().reshape(cache_rows, width)
    for r in range(rows):
        if int(idx[r]) != SENTINEL:
            ref[int(idx[r])] = src[r]
    got = ttnn.to_torch(t_cache).reshape(cache_rows, width)
    assert torch.equal(got, ref)


def test_fp32_to_uint32_sentinel(device):
    """The DSA index builder computes indices in fp32 and relies on typecast fp32 -> uint32 saturating 2**32 to
    0xFFFFFFFF (the sparse_sdpa / scatter_rows sentinel)."""
    x = torch.tensor([[0.0, 5.0, 1048575.0, 4294967296.0]])
    t = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    u = ttnn.to_torch(ttnn.typecast(t, ttnn.uint32)).to(torch.int64)
    assert u.flatten().tolist()[:3] == [0, 5, 1048575]
    assert u.flatten().tolist()[3] == SENTINEL, u
