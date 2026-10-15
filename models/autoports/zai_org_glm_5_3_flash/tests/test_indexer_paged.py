# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.indexer_score_dsa with page_table: key tile rows (32 pooled keys = one 128-token page block) read
through a permuted, non-contiguous block table must score exactly like the contiguous per-slot key cache."""

import pytest
import torch

import ttnn

HI, D = 32, 128


@pytest.mark.parametrize("P,rows", [(1024 + 32, 128), (4096 + 32, 256)])
def test_indexer_score_paged(device, P, rows):
    gen = torch.Generator().manual_seed(P)
    slots, slot = 3, 1
    nblk = P // 32
    pool_blocks = slots * nblk + 5
    keys = torch.randn(slots, P, D, generator=gen).bfloat16()
    perm = torch.randperm(pool_blocks, generator=gen)[: slots * nblk].reshape(slots, nblk)
    pool = torch.zeros(pool_blocks * 32, D).bfloat16()
    for s in range(slots):
        for j in range(nblk):
            pool[perm[s, j] * 32 : (perm[s, j] + 1) * 32] = keys[s, j * 32 : (j + 1) * 32]
    # the paged pool, shaped [B', 1, P, D] (B' * P >= pool keys) so the scored key length T = P
    bp = (pool_blocks * 32 + P - 1) // P
    pool = torch.cat([pool, torch.zeros(bp * P - pool.shape[0], D).bfloat16()]).reshape(bp, 1, P, D)
    tile = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    q = tile(torch.randn(1, HI, rows, D, generator=gen) * 0.1)
    w = tile(torch.randn(1, 1, rows, HI, generator=gen) * 0.2)
    pt = ttnn.from_torch(perm.to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    chunk_start = P - 32  # every real key visible to every row
    ref = ttnn.experimental.indexer_score_dsa(
        q, tile(keys.reshape(slots, 1, P, D)), w, chunk_start_idx=chunk_start, cache_batch_idx=slot
    )
    got = ttnn.experimental.indexer_score_dsa(
        q, tile(pool), w, chunk_start_idx=chunk_start, cache_batch_idx=slot, page_table=pt
    )
    r, g = ttnn.to_torch(ref).float(), ttnn.to_torch(got).float()
    fin = torch.isfinite(r)
    assert torch.equal(fin, torch.isfinite(g))
    d = (r[fin] - g[fin]).abs().max().item()
    print(f"P={P} rows={rows}: paged vs contiguous max|diff| {d:.3e} ({int(fin.sum())} finite scores)")
    assert d == 0.0
