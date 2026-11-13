# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Rows-mode attention at long context on one die (MiniMax-M2.7 per-die shapes: 12 q heads, 2 kv heads, head dim 128,
bfp8 paged KV, block 64): paged SDPA decode for 1 and 4 users (the verify rows today: every row reads the whole KV)
vs the chunked prefill SDPA over one 128-row chunk at an aligned start (reads the KV once). Trace-replay time per call.
usage (in the image, run_in_image.sh): POC_CTX=<context> (default 40000)"""
import os
import time

import torch

import ttnn

CTX = int(os.environ.get("POC_CTX", "40000"))
NH, NKV, HD, BLK = 12, 2, 128, 64


def bench(dev, fn, reps=10, iters=10):
    fn()
    ttnn.synchronize_device(dev)
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


dev = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=64 << 20)
try:
    nblk = (CTX + 256) // BLK + 1
    mk = lambda: ttnn.from_torch(
        torch.randn(nblk, NKV, BLK, HD) * 0.5,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,  # noqa: E731
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    K, V = mk(), mk()
    grid = dev.compute_with_storage_grid_size()
    res = {}
    for B in (1, 4):
        pt = ttnn.from_torch(
            torch.arange(nblk, dtype=torch.int32).repeat(B, 1),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        pos = ttnn.from_torch(
            torch.tensor([CTX + j for j in range(B)], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q = ttnn.from_torch(
            torch.randn(1, B, NH, HD),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y), exp_approx_mode=False, q_chunk_size=0, k_chunk_size=0
        )

        def dec():
            o = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q,
                K,
                V,
                page_table_tensor=pt,
                cur_pos_tensor=pos,
                scale=HD**-0.5,
                program_config=cfg,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(o)

        res[f"decode B={B}"] = bench(dev, dec)
    # chunked prefill: one 128-row chunk starting at an aligned position <= CTX (rows past CTX are padding)
    start = (CTX // 128) * 128
    pt1 = ttnn.from_torch(
        torch.arange(nblk, dtype=torch.int32).reshape(1, -1),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    for qc, kc in ((32, 128), (64, 256), (128, 512)):
        qp = ttnn.from_torch(
            torch.randn(1, NH, 128, HD),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        pcfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y), exp_approx_mode=False, q_chunk_size=qc, k_chunk_size=kc
        )

        def chk():
            o = ttnn.transformer.chunked_scaled_dot_product_attention(qp, K, V, pt1, start, program_config=pcfg)
            ttnn.deallocate(o)

        try:
            res[f"chunked 128 rows q{qc} k{kc}"] = bench(dev, chk)
        except Exception as e:  # noqa: BLE001
            res[f"chunked 128 rows q{qc} k{kc}"] = f"failed: {str(e)[:120]}"
    kv_mb = 2 * CTX * NKV * HD * 1.0625 / 1e6
    print(f"SDPA_ROWS context {CTX} (KV {kv_mb:.1f} MB per layer per die):", flush=True)
    for k_, v in res.items():
        print(f"  {k_}: {v if isinstance(v, str) else f'{v:.1f} us'}", flush=True)
finally:
    ttnn.close_device(dev)
