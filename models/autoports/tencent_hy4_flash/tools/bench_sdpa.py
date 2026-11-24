# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark: sparse_sdpa for one decode token, q [1, 32, T, 608] RM with T = 32 (the padded pass) vs T = 1..8
(only the real rows); exactness of the real rows and device time from trace replays."""

from __future__ import annotations

import torch
import ttnn

from models.autoports.tencent_hy4_flash.tools.bench_dsh import timed
from models.autoports.tencent_hy4_flash.tt.attention import K_CHUNK, SENTINEL, SKF


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100_000_000)
    try:
        rep = ttnn.ReplicateTensorToMesh(mesh)
        torch.manual_seed(0)
        rows_cache, W, H = 4096, 608, 32
        cache = torch.randn(1, 1, rows_cache, W).to(torch.bfloat16)
        ct = ttnn.from_torch(cache, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
        for nkeys in (513, 2049):
            K = -(-nkeys // K_CHUNK) * K_CHUNK
            q = torch.randn(1, H, 32, W).to(torch.bfloat16)
            idx = torch.full((1, 1, 32, K), SENTINEL, dtype=torch.int64)
            keys = torch.arange(1000, 1000 + nkeys)
            idx[..., :nkeys] = keys
            i32 = idx.to(torch.uint32).view(torch.int32) if hasattr(torch, "uint32") else idx.to(torch.int32)
            res = {}
            for T in (32, 8, 1):
                qt = ttnn.from_torch(q[:, :, :T].contiguous(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                     device=mesh, mesh_mapper=rep)
                it = ttnn.from_torch(idx[:, :, :T].to(torch.int32).contiguous(), dtype=ttnn.uint32,
                                     layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
                try:
                    t, o = timed(mesh, lambda: ttnn.transformer.sparse_sdpa(qt, ct, it, 512, kv_format=SKF.BF16,
                                                                           scale=1 / 16, k_chunk_size=K_CHUNK))
                    res[T] = (t, ttnn.to_torch(ttnn.get_device_tensors(o)[0]).float()[:, :, :1])
                    print(f"keys {nkeys} T {T:2d}: {t:6.1f} us, out {list(o.shape)}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"keys {nkeys} T {T:2d}: {str(e)[:200]}", flush=True)
            if 32 in res:
                for T, (t, o) in res.items():
                    print(f"  T {T}: row 0 equal to T=32's: {torch.equal(o, res[32][1])}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
