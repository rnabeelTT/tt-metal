# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Rows top-k (models/demos/blackhole/qwen36/tt/decode_sampling/ops.py) on the (1, 4) mesh vs torch: per device, rows
0 .. B-1 of [32, 62080] BF16 logits, K = 32, sorted by value then lower index; rows >= B copy row 0; traced timing vs
the sampler's ttnn.pad + ttnn.topk.

    python test_rows_topk.py
"""

import time

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.decode_sampling.ops import RowsTopK

W, K, ND = 62080, 32, 4


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, ND), trace_region_size=32 << 20)
    try:
        op = RowsTopK(mesh, W, K)
        comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
        for B in (1, 2, 8):
            lg = (torch.randn(ND, 1, 32, W) * 3).bfloat16()
            lg[0, 0, 0, 1000] = lg[0, 0, 0, 2000] = 50.0  # an exact tie at the top: lower index first
            x = ttnn.from_torch(lg, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
            v, i = op(x, B)
            gv = ttnn.to_torch(v, mesh_composer=comp).float()
            gi = ttnn.to_torch(i, mesh_composer=comp).long()
            for d in range(ND):
                for r in range(32):
                    src = min(r, 0) if r >= B else r
                    row = lg[d, 0, src].float()
                    order = sorted(range(W), key=lambda c: (-row[c].item(), c))[:K] if r < 2 else None
                    if order is not None:
                        assert torch.equal(gi[d, 0, r], torch.tensor(order)), (B, d, r, gi[d, 0, r][:8].tolist(), order[:8])
                        assert torch.equal(gv[d, 0, r], row[order]), (B, d, r)
                    else:  # rows 2..: values only (sorted top-K set)
                        ref = torch.sort(row, descending=True).values[:K]
                        assert torch.equal(gv[d, 0, r], ref), (B, d, r)
            print(f"B={B}: ok on {ND} devices", flush=True)
            for t in (x, v, i):
                ttnn.deallocate(t)
        # traced timing vs the sampler's pad + ttnn.topk
        x = ttnn.from_torch(torch.randn(1, 1, 32, W).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        def ours():
            v, i = op(x, 1)
            ttnn.deallocate(v)
            ttnn.deallocate(i)

        def stock():
            xp = ttnn.pad(x, [(0, 0), (0, 0), (0, 0), (0, 65536 - W)], value=-3.0e38)
            v, i = ttnn.topk(xp, k=K, dim=-1, stable=False)
            for t in (xp, v, i):
                ttnn.deallocate(t)

        for label, fn in (("rows top-k B=1", ours), ("pad + ttnn.topk", stock)):
            fn()
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            for _ in range(10):
                fn()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            t0 = time.perf_counter()
            for _ in range(20):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            print(f"traced {label}: {(time.perf_counter() - t0) / 200 * 1e6:.1f} us per call")
            ttnn.release_trace(mesh, tid)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
