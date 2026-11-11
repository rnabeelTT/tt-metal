# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Microbenchmark: the rows-mode K/V cache writes of one attention layer at MiniMax-M2.7's per-die shapes (4 verify
rows of one user, 2 KV heads, head dim 128, bfp8 paged cache of 64-token blocks): the 4 masked
paged_fused_update_cache calls the verify issues (tp._forward_rows_ops, one row per call) vs a single call (its timing
only: rows sharing a cache tile race in one call). Trace-replay time per layer. usage (in the image): python
bench_kvupd.py"""

import time

import torch

import ttnn

B, NKV, HD, BLOCK, NBLK = 4, 2, 128, 64, 64


def main():
    dev = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=32 << 20)
    dram = ttnn.DRAM_MEMORY_CONFIG

    def shard(row):
        return ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, HD),
            core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, row), ttnn.CoreCoord(B - 1, row))}),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    cache = lambda: ttnn.from_torch(  # noqa: E731
        torch.zeros(NBLK, NKV, BLOCK, HD), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=dram
    )
    kc, vc = cache(), cache()
    kv = lambda r: ttnn.from_torch(  # noqa: E731
        torch.randn(1, B, NKV, HD), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=shard(r)
    )
    k, v = kv(0), kv(1)
    pt = ttnn.from_torch(
        torch.arange(NBLK, dtype=torch.int32).reshape(1, NBLK).repeat(B, 1),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=dev,
        memory_config=dram,
    )
    p0 = 1000
    rows = []
    for b in range(B):
        idx = torch.full((B,), -1, dtype=torch.int32)
        idx[b] = p0 + b
        rows.append(
            ttnn.from_torch(idx, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=dram)
        )
    allpos = ttnn.from_torch(
        torch.arange(p0, p0 + B, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=dev,
        memory_config=dram,
    )

    def four():
        for b in range(B):
            ttnn.experimental.paged_fused_update_cache(kc, k, vc, v, update_idxs_tensor=rows[b], page_table=pt)

    def one():
        ttnn.experimental.paged_fused_update_cache(kc, k, vc, v, update_idxs_tensor=allpos, page_table=pt)

    def bench(fn, reps=50, iters=20):
        fn()
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        for _ in range(reps):
            fn()
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        t0 = time.time()
        for _ in range(iters):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        ttnn.release_trace(dev, tid)
        return (time.time() - t0) / (reps * iters) * 1e6

    print(f"KVUPD 4 masked calls {bench(four):.1f} us per layer, 1 call {bench(one):.1f} us", flush=True)
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
