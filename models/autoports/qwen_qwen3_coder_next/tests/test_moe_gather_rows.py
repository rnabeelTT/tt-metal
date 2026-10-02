# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.moe_gather_rows (multi-core tile gather of the MoE dispatch): real groups' rows match torch's gather and the
embedding-op gather it replaces (gather_rows), padding groups are skipped, odd group counts split across the core
groups, and the op is faster than the embedding gather at the layer's shapes."""
import time

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_coder_next.tt.moe_dispatch import gather_rows

H = 3072
E = 256


def _dispatch_like(T, k, G, gs=32, seed=0):
    """A sorted-dispatch-like layout: real groups first (expert-ascending, each group 32 rows whose source tokens are
    real tokens of that expert, padded rows pointing at token 0), then sentinel padding groups."""
    torch.manual_seed(seed)
    ids = torch.randint(0, E, (T, k))
    pairs = sorted((int(ids[t, j]), t) for t in range(T) for j in range(k))
    groups, cur, cur_e = [], [], None
    for e, t in pairs:
        if e != cur_e or len(cur) == gs:
            if cur:
                groups.append((cur_e, cur))
            cur, cur_e = [], e
        cur.append(t)
    if cur:
        groups.append((cur_e, cur))
    assert len(groups) <= G, (len(groups), G)
    src = torch.zeros(G * gs, dtype=torch.int64)
    gid = torch.full((G,), 0xFFFF, dtype=torch.int64)
    for g, (e, toks) in enumerate(groups):
        gid[g] = e
        src[g * gs : g * gs + len(toks)] = torch.tensor(toks)
    return src, gid, len(groups)


@pytest.mark.parametrize(
    "T,k,G", [(16, 8, 129), (2048, 8, 761), (37, 8, 281)], ids=["t16_g129", "t2048_g761", "t37_g281"]
)
def test_moe_gather_rows_matches_torch(device, T, k, G):
    src, gid, n_real = _dispatch_like(T, k, G)
    x = torch.randn(T, H, dtype=torch.bfloat16)
    x_t = ttnn.from_torch(x.reshape(1, 1, T, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    x_rm = ttnn.to_layout(ttnn.reshape(x_t, (T, H)), ttnn.ROW_MAJOR_LAYOUT)
    src_t = ttnn.from_torch(
        src.to(torch.int32).reshape(1, G * 32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    gid_t = ttnn.from_torch(
        gid.to(torch.int32).reshape(1, G), dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    out = ttnn.moe_gather_rows(x_rm, src_t, gid_t, E, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert out.shape == (1, G, 32, H), out.shape
    got = ttnn.to_torch(out).reshape(G, 32, H)
    ref = x[src].reshape(G, 32, H)
    assert torch.equal(got[:n_real].float(), ref[:n_real].float()), "real groups differ from torch gather"
    # same values as the embedding-op gather the layer used (real groups)
    old = ttnn.to_torch(gather_rows(x_t, src_t, G * 32)).reshape(G, 32, H)
    assert torch.equal(got[:n_real].float(), old[:n_real].float())
    print(f"MOE_GATHER_ROWS T={T} G={G} real={n_real}: bit-exact vs torch and the embedding gather")


@pytest.mark.parametrize("T,G", [(16, 129), (2048, 761)], ids=["t16", "t2048"])
def test_moe_gather_rows_faster_than_embedding(device, T, G):
    src, gid, n_real = _dispatch_like(T, 8, G)
    x = torch.randn(T, H, dtype=torch.bfloat16)
    x_t = ttnn.from_torch(x.reshape(1, 1, T, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    src_t = ttnn.from_torch(
        src.to(torch.int32).reshape(1, G * 32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    gid_t = ttnn.from_torch(
        gid.to(torch.int32).reshape(1, G), dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    def run_new():
        x_rm = ttnn.to_layout(ttnn.reshape(x_t, (T, H)), ttnn.ROW_MAJOR_LAYOUT)
        o = ttnn.moe_gather_rows(x_rm, src_t, gid_t, E, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x_rm)
        return o

    def run_old():
        return gather_rows(x_t, src_t, G * 32)

    res = {}
    for name, fn in (("embedding", run_old), ("kernel", run_new)):
        for _ in range(3):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        for _ in range(10):
            o = fn()
            ttnn.deallocate(o)
        ttnn.synchronize_device(device)
        res[name] = (time.perf_counter() - t0) / 10 * 1e3
    print(
        f"MOE_GATHER_ROWS_PERF T={T} G={G} real={n_real}: embedding {res['embedding']:.3f} ms, kernel {res['kernel']:.3f} ms (host wall incl. sync)"
    )
    assert res["kernel"] < res["embedding"], res
