# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Device test: write one expert into slot s of a stacked bfp4 expert tensor [1, C, H, W] (sharded over the mesh on
the last dim) with ttnn.fill_cache through a [C, 1, H, W] view; check only slot s changed. Also times the write."""

import sys
import time

import torch

import ttnn

import os

STAGE_DT = getattr(ttnn, os.environ.get("STAGE_DT", "bfloat16"))


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    try:
        C, H, W = 16, 4096, 4 * 512  # W per device 512 (= 2 * Nl for the 397B)
        sh = ttnn.ShardTensorToMesh(mesh, dim=-1)
        base = torch.randn(1, C, H, W, dtype=torch.bfloat16)
        dev = ttnn.from_torch(base, dtype=getattr(ttnn, os.environ.get("CACHE_DT", "bfloat8_b")), layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=sh,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)
        before = ttnn.to_torch(dev, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()
        new = torch.randn(1, 1, H, W, dtype=torch.bfloat16) * 3
        stage = ttnn.from_torch(new, dtype=STAGE_DT, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=sh,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
        shp = [int(v) for v in dev.shape]  # per-device shape on a mesh
        view = ttnn.reshape(dev, [shp[1], 1, shp[2], shp[3]])
        print("dev shape", shp, "view", list(view.shape), flush=True)
        s = 5
        ttnn.fill_cache(view, stage, s)
        ttnn.synchronize_device(mesh)
        after = ttnn.to_torch(dev, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()
        changed = [(after[0, i] != before[0, i]).any().item() for i in range(C)]
        exp = ttnn.to_torch(stage, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()
        ok_slot = torch.equal(after[0, s], exp[0, 0]) or (after[0, s] - new[0, 0].float()).norm().item() / new[0, 0].float().norm().item() < 0.2
        print("slot rel err vs staged bf16", ((after[0, s] - new[0, 0].float()).norm() / new[0, 0].float().norm()).item(), flush=True)
        print("changed slots:", [i for i, c in enumerate(changed) if c], "slot matches staged:", ok_slot, flush=True)
        # timing: host conversion + upload + fill
        host = ttnn.from_torch(new, dtype=STAGE_DT, layout=ttnn.TILE_LAYOUT, mesh_mapper=sh)
        t0 = time.perf_counter()
        for _ in range(10):
            h2 = ttnn.from_torch(new, dtype=STAGE_DT, layout=ttnn.TILE_LAYOUT, mesh_mapper=sh)
        t1 = time.perf_counter()
        for i in range(10):
            ttnn.copy_host_to_device_tensor(host, stage)
            ttnn.fill_cache(view, stage, i % C)
        ttnn.synchronize_device(mesh)
        t2 = time.perf_counter()
        print(f"host bfp4 convert {1e3 * (t1 - t0) / 10:.1f} ms/tensor; upload+fill {1e3 * (t2 - t1) / 10:.2f} ms/tensor",
              flush=True)
        print("RESULT", "PASS" if ok_slot and changed == [i == s for i in range(C)] else "FAIL")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
