# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark (traced replays): the sparse-regime decode sparse_sdpa, q [1, 32, 32, 608] RM, 2048 selected keys + the
tail, over a 128K-row cache: real rows R (the rest all-sentinel, HY4_IDX_PADSENT) or every row with keys (before),
contiguous vs scattered keys, key chunk 128 / 256 / 512 (index width padded with sentinels), fewer keys per row (what
a key split across cores would leave each core)."""

from __future__ import annotations

import torch
import ttnn

from models.autoports.tencent_hy4_flash.tools.bench_dsh import timed
from models.autoports.tencent_hy4_flash.tt.attention import SENTINEL, SKF


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100_000_000)
    try:
        rep = ttnn.ReplicateTensorToMesh(mesh)
        torch.manual_seed(0)
        rows_cache, W, H = 131072 + 256, 608, 32
        ct = ttnn.from_torch(torch.randn(1, 1, rows_cache, W).to(torch.bfloat16), dtype=ttnn.bfloat16,
                             layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
        q = torch.randn(1, H, 32, W).to(torch.bfloat16)
        qt = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
        g = torch.Generator().manual_seed(1)
        for nkeys, layout in ((2049, "scattered"), (2049, "contiguous"), (1025, "scattered"), (513, "scattered"),
                              (257, "scattered")):
            if layout == "scattered":
                keys = torch.sort(torch.randperm(131072, generator=g)[:nkeys]).values
            else:
                keys = torch.arange(100000, 100000 + nkeys)
            for kc in (128, 256, 512):
                K = -(-(nkeys + 4) // kc) * kc
                for real in (1, 5, 32):
                    idx = torch.full((1, 1, 32, K), SENTINEL, dtype=torch.int64)
                    idx[..., :nkeys] = keys
                    idx[:, :, real:, :] = SENTINEL
                    idx[:, :, real:, 0] = rows_cache - 1  # (the padding rows' sink, HY4_PAD_SINK)
                    it = ttnn.from_torch(idx.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                         device=mesh, mesh_mapper=rep)
                    try:
                        t, o = timed(mesh, lambda: ttnn.transformer.sparse_sdpa(qt, ct, it, 512, kv_format=SKF.BF16,
                                                                               scale=1 / 16, k_chunk_size=kc), n=100)
                        print(f"SDPA_ROWS keys {nkeys} {layout} kc {kc} real rows {real}: {t:6.1f} us", flush=True)
                        ttnn.deallocate(o)
                    except Exception as e:  # noqa: BLE001
                        print(f"SDPA_ROWS keys {nkeys} {layout} kc {kc} real rows {real}: {str(e)[:160]}", flush=True)
                    ttnn.deallocate(it)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
