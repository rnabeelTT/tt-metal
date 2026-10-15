# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.transformer.sparse_sdpa with a page table (page_table / page_block_size): logical token indices remapped in the
reader through a permuted, non-contiguous block table must give the same output as the contiguous per-slot cache.

Prefill form: every query row reads the page-table row cache_batch_idx. Decode form: query row b reads row b."""

import pytest
import torch

import ttnn

H, R, V_DIM, TOPK, BS = 64, 512, 512, 2080, 128
SENTINEL = 0xFFFFFFFF


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def _indices(rows, lengths, gen):
    """Logical index rows: a random subset of [0, length) (sorted) then sentinels; lengths per row."""
    idx = torch.full((rows, TOPK), SENTINEL, dtype=torch.int64)
    for r, n in enumerate(lengths):
        k = min(n, TOPK - 3)
        sel = torch.randperm(n, generator=gen)[:k].sort().values
        idx[r, :k] = sel
    return idx


def _rm(t, dtype, device):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def _run(device, q, kv, idx, **kw):
    out = ttnn.transformer.sparse_sdpa(
        _rm(q, ttnn.bfloat16, device),
        kv,
        _rm(idx.to(torch.int64).reshape(1, 1, *idx.shape), ttnn.uint32, device),
        V_DIM,
        kv_format=ttnn.SparseKVFormat.BF16,
        scale=0.0625,
        k_chunk_size=32,
        **kw,
    )
    return ttnn.to_torch(out).float()


@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_sparse_sdpa_paged(device, mode):
    gen = torch.Generator().manual_seed(1)
    slots, T = 4, 8192  # per-slot capacity (logical)
    nblk = T // BS
    total_blocks = slots * nblk + 7
    # contiguous reference cache [slots, 1, T, R] and its paged copy under a random permutation of blocks
    lat = torch.randn(slots, T, R, generator=gen).bfloat16()
    perm = torch.randperm(total_blocks, generator=gen)[: slots * nblk].reshape(slots, nblk)
    pool = torch.zeros(total_blocks * BS, R).bfloat16()
    for s in range(slots):
        for j in range(nblk):
            pool[perm[s, j] * BS : (perm[s, j] + 1) * BS] = lat[s, j * BS : (j + 1) * BS]
    pt = _rm(perm.to(torch.int32), ttnn.int32, device)
    kv_paged = _rm(pool.reshape(1, 1, -1, R), ttnn.bfloat16, device)
    if mode == "prefill":
        rows, slot = 64, 2
        q = torch.randn(1, H, rows, R, generator=gen) * 0.1
        idx = _indices(rows, [3000 + 70 * r for r in range(rows)], gen)
        kv_ref = _rm(lat.reshape(slots, 1, T, R), ttnn.bfloat16, device)
        ref = _run(device, q, kv_ref, idx, cache_batch_idx=slot)
        got = _run(device, q, kv_paged, idx, cache_batch_idx=slot, page_table=pt, page_block_size=BS)
    else:
        rows = slots  # one query row per slot
        q = torch.randn(1, H, rows, R, generator=gen) * 0.1
        idx = _indices(rows, [8000, 5000, 129, 2051], gen)
        flat = idx.clone()  # contiguous reference: flat rows slot * T + t
        for b in range(rows):
            ok = flat[b] != SENTINEL
            flat[b, ok] += b * T
        kv_ref = _rm(lat.reshape(1, 1, slots * T, R), ttnn.bfloat16, device)
        ref = _run(device, q, kv_ref, flat)
        got = _run(device, q, kv_paged, idx, page_table=pt, page_block_size=BS)
    p, e = _pcc(got, ref), (got - ref).abs().max().item()
    print(f"{mode}: paged vs contiguous PCC {p:.7f} max|diff| {e:.3e}")
    assert torch.equal(got, ref) or p > 0.99999
