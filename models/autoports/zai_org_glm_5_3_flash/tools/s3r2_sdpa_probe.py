# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-3 re-review #2 P2-4: DSA sparse decode attention past position 2048 on one die (ttnn.transformer.sparse_sdpa,
64 heads, latent 512, top-k 2080 indices, paged bf16 latent cache, decode form).

The deep-position Tracy (candidates_r2/deep_pos_16384) shows SparseSDPAOperation 563 us at 1 user and 574 us at 32 users:
the op splits its work over query rows only (sparse_sdpa_program_factory.cpp: base = S / num_cores), so one user's 64
heads run on ONE core. Candidates measured here (traced, mean of 30 replays, outputs compared bit-exactly / PCC):
  k_chunk_size 32 (the model's) / 64 / 128 at 1 and 32 users
  1 user, head split: the 64 heads as 2 query rows of 32 heads ([1,64,1,512] -> [1,32,2,512], one index row per query
  row, cache_batch_idx = the slot: both rows read the user's page-table row), output reshaped back

usage: python -m models.autoports.zai_org_glm_5_3_flash.tools.s3r2_sdpa_probe"""

import time

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.harness import open_mesh, pcc

H, R, TOPK, BS = 64, 512, 2080, 128
SENT = 0xFFFFFFFF


def traced(mesh, fn, n=30):
    o = fn()
    ttnn.synchronize_device(mesh)
    ttnn.deallocate(o)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    t = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    us = (time.perf_counter() - t) / n * 1e6
    y = ttnn.to_torch(out).float()
    ttnn.release_trace(mesh, tid)
    ttnn.deallocate(out)
    return y, us


def rm(t, dt, mesh):
    return ttnn.from_torch(t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)


def main():
    mesh = open_mesh()
    g = torch.Generator().manual_seed(0)
    try:
        for S, T in ((1, 16384), (32, 16384)):
            nblk = T // BS
            pool = (torch.randn(S * nblk * BS, R, generator=g) * 0.5).bfloat16()
            kv = rm(pool.reshape(1, 1, -1, R), ttnn.bfloat16, mesh)
            pt = rm(torch.arange(S * nblk, dtype=torch.int32).reshape(S, nblk), ttnn.int32, mesh)
            idx = torch.stack([torch.randperm(T - 1, generator=g)[:TOPK].sort().values for _ in range(S)])  # [S, TOPK]
            tidx = rm(idx.to(torch.int64).reshape(1, 1, S, TOPK), ttnn.uint32, mesh)
            q = torch.randn(1, H, S, R, generator=g) * 0.1
            tq = rm(q.bfloat16(), ttnn.bfloat16, mesh)

            def run(kc, q_=tq, idx_=tidx, **kw):
                return lambda: ttnn.transformer.sparse_sdpa(
                    q_,
                    kv,
                    idx_,
                    R,
                    kv_format=ttnn.SparseKVFormat.BF16,
                    scale=0.0625,
                    k_chunk_size=kc,
                    page_table=pt,
                    page_block_size=BS,
                    **kw,
                )

            ref = None
            for kc in (32, 64, 128):
                try:
                    y, us = traced(mesh, run(kc))
                    if ref is None:
                        ref = y
                    print(f"S={S:2d} T={T} k_chunk {kc:3d}: {us:8.1f} us  PCC vs k32 {pcc(y, ref):.6f}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"S={S:2d} k_chunk {kc}: ERROR {str(e).splitlines()[0][:200]}", flush=True)
            if S == 1:
                # head split: [1,64,1,512] -> [1,32,2,512] (query row j = heads 32j..32j+31), index row duplicated
                q2 = q.reshape(1, 2, 32, R).permute(0, 2, 1, 3).contiguous()  # [1,32,2,R]: q2[0,h,j] = q[0,32j+h,0]
                tq2 = rm(q2.bfloat16(), ttnn.bfloat16, mesh)
                tidx2 = rm(idx.to(torch.int64).repeat(2, 1).reshape(1, 1, 2, TOPK), ttnn.uint32, mesh)
                for kc in (32, 64, 128):
                    try:
                        y2, us = traced(mesh, run(kc, tq2, tidx2, cache_batch_idx=0))
                        y2b = y2.reshape(1, 32, 2, -1).permute(0, 2, 1, 3).reshape(1, H, 1, -1)
                        print(
                            f"S= 1 head split (2 rows x 32 heads, cache_batch_idx) k_chunk {kc:3d}: {us:8.1f} us  "
                            f"PCC vs k32 {pcc(y2b, ref):.6f} exact {torch.equal(y2b, ref)}",
                            flush=True,
                        )
                    except Exception as e:  # noqa: BLE001
                        print(f"S= 1 head split k_chunk {kc}: ERROR {str(e).splitlines()[0][:200]}", flush=True)
            for t in (kv, pt, tidx, tq):
                ttnn.deallocate(t)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
