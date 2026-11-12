# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Test + microbenchmark of attn_decode/kvrows.py (the multi-row paged KV update, one generic_op) against the n masked
paged_fused_update_cache calls it replaces, at MiniMax-M2.7's per-die shapes (2 KV heads, head dim 128, bfp8 paged
cache of 64-token blocks): the caches must match bit for bit (positions inside one tile, across a tile boundary, across
a block boundary, with skipped rows), then trace-replay time per layer for n = 4 and 8.
usage (in the image): python bench_kvrows.py"""

import sys
import time

sys.path.insert(0, "/opt/tt-metal")
import torch  # noqa: E402

import ttnn  # noqa: E402
from models.demos.blackhole.qwen36.tt.attn_decode.kvrows import kv_rows_update  # noqa: E402

NKV, HD, BLOCK, NBLK = 2, 128, 64, 64


def main():
    dev = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=32 << 20)
    dram = ttnn.DRAM_MEMORY_CONFIG

    def shard(B, row):
        return ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, HD),
            core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, row), ttnn.CoreCoord(B - 1, row))}),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    g = torch.Generator().manual_seed(0)
    base_k = torch.randn(NBLK, NKV, BLOCK, HD, generator=g)
    base_v = torch.randn(NBLK, NKV, BLOCK, HD, generator=g)
    perm = torch.randperm(NBLK, generator=g).to(torch.int32)

    def cache(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=dram)

    def i32(t):
        return ttnn.from_torch(t, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=dram)

    ok = True
    for B in (4, 8):
        pt = i32(perm.reshape(1, NBLK).repeat(B, 1))
        k = ttnn.from_torch(
            torch.randn(1, B, NKV, HD, generator=g),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=shard(B, 0),
        )
        v = ttnn.from_torch(
            torch.randn(1, B, NKV, HD, generator=g),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=shard(B, 1),
        )
        for p0, skip in ((1000, ()), (30, ()), (62, ()), (125, (1,)), (200, (B - 1,)), (7, tuple(range(B)))):
            m = torch.full((B, B), -1, dtype=torch.int32)
            for b in range(B):
                if b not in skip:
                    m[b, b] = p0 + b
            rows = [i32(m[b].clone()) for b in range(B)]
            idx = i32(m)
            ka, va, kb, vb = cache(base_k), cache(base_v), cache(base_k), cache(base_v)
            for b in range(B):
                ttnn.experimental.paged_fused_update_cache(ka, k, va, v, update_idxs_tensor=rows[b], page_table=pt)
            kv_rows_update(kb, k, vb, v, idx, pt)
            same = all(torch.equal(ttnn.to_torch(x), ttnn.to_torch(y)) for x, y in ((ka, kb), (va, vb)))
            changed = not torch.equal(ttnn.to_torch(ka), ttnn.to_torch(cache(base_k)))
            print(f"B {B} p0 {p0} skip {skip}: identical {same} (cache changed {changed})", flush=True)
            ok &= same

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

        m = torch.full((B, B), -1, dtype=torch.int32)
        for b in range(B):
            m[b, b] = 1000 + b
        rows = [i32(m[b].clone()) for b in range(B)]
        idx = i32(m)
        kc, vc = cache(base_k), cache(base_v)
        t_old = bench(
            lambda: [
                ttnn.experimental.paged_fused_update_cache(kc, k, vc, v, update_idxs_tensor=r, page_table=pt)
                for r in rows
            ]
        )
        t_new = bench(lambda: kv_rows_update(kc, k, vc, v, idx, pt))
        print(f"B {B}: {B} masked calls {t_old:.1f} us, kv_rows_update {t_new:.1f} us", flush=True)
    print("KVROWS ALL IDENTICAL" if ok else "KVROWS MISMATCH", flush=True)
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
