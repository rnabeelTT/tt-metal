# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.dsa_decode_scores: indexer scores of one decode query per slot over a paged pooled-key cache
(permuted, non-contiguous 32-key blocks), vs torch. Keys t < limits[s] must match; blocks starting at or past
limits[s] must be -inf."""

import pytest
import torch

import ttnn

HI, D = 32, 128


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


@pytest.mark.parametrize("S,nblk", [(1, 40), (8, 33), (32, 64)])
def test_dsa_decode_scores(device, S, nblk):
    gen = torch.Generator().manual_seed(S * 1000 + nblk)
    P = nblk * 32
    pool_blocks = S * nblk + 3
    keys = torch.randn(S, P, D, generator=gen).bfloat16()
    perm = torch.randperm(pool_blocks, generator=gen)[: S * nblk].reshape(S, nblk)
    pool = torch.zeros(pool_blocks * 32, D).bfloat16()
    for s in range(S):
        for j in range(nblk):
            pool[perm[s, j] * 32 : (perm[s, j] + 1) * 32] = keys[s, j * 32 : (j + 1) * 32]
    q = (torch.randn(S * HI, D, generator=gen) * 0.1).bfloat16()
    w = torch.randn(S, HI, generator=gen) * 0.2
    limits = torch.randint(1, P + 1, (S,), generator=gen)
    limits[0] = P  # full width
    if S > 2:
        limits[1] = 0  # empty slot: every block -inf
        limits[2] = 33  # one key into the second block
    ref = torch.einsum("shd,std->sht", q.float().reshape(S, HI, D), keys.float()).relu()
    ref = torch.einsum("sh,sht->st", w, ref)
    tile = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    rm = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out = ttnn.experimental.dsa_decode_scores(
        tile(q.reshape(1, 1, S * HI, D), ttnn.bfloat16),
        tile(w.reshape(1, 1, S, HI), ttnn.float32),
        tile(pool.reshape(1, 1, -1, D), ttnn.bfloat16),
        rm(perm.to(torch.int32), ttnn.int32),
        rm(limits.to(torch.int64).reshape(1, S), ttnn.uint32),
        nblk,
    )
    got = ttnn.to_torch(out).float().reshape(S, P)
    worst = 1.0
    for s in range(S):
        n = int(limits[s])
        if n:
            ps = _pcc(got[s, :n], ref[s, :n])
            if ps < 0.9999:
                print(f"slot {s} limit {n}: PCC {ps:.4f}")
            worst = min(worst, ps)
        first_dead = ((n + 31) // 32) * 32  # blocks starting at or past the limit
        assert torch.isneginf(got[s, first_dead:]).all(), f"slot {s}: blocks past limit {n} not -inf"
    print(f"S={S} nblk={nblk}: min per-slot PCC over valid keys {worst:.7f}")
    assert worst > 0.9999
