# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host <-> device transfer costs on the (1, 4) mesh for the small per-round tensors of the DFlash step (replicated):
copy_host_to_device_tensor of row-major uint32 / tile bf16 / tile fp32, and to_torch reads, idle device vs behind queued
work."""
import time

import torch

import ttnn


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), l1_small_size=24576)
    rep = ttnn.ReplicateTensorToMesh(mesh)
    cases = {
        "u32 RM [1,32]": (torch.zeros(1, 32, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
        "bf16 tile [1,1,32,128]": (torch.zeros(1, 1, 32, 128), ttnn.bfloat16, ttnn.TILE_LAYOUT),
        "bf16 tile [1,1,1,2304]": (torch.zeros(1, 1, 1, 2304), ttnn.bfloat16, ttnn.TILE_LAYOUT),
        "fp32 tile [1,1,1,32]": (torch.zeros(1, 1, 1, 32), ttnn.float32, ttnn.TILE_LAYOUT),
    }
    for name, (t, dt, lay) in cases.items():
        dev = ttnn.from_torch(t, dtype=dt, layout=lay, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep)
        for _ in range(3):
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(t, dtype=dt, layout=lay), dev)
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(50):
            h = ttnn.from_torch(t, dtype=dt, layout=lay)
            ttnn.copy_host_to_device_tensor(h, dev)
        t1 = time.perf_counter()
        ttnn.synchronize_device(mesh)
        t2 = time.perf_counter()
        for _ in range(20):
            ttnn.to_torch(dev, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        t3 = time.perf_counter()
        print(f"HOSTIO {name}: write {(t1 - t0) / 50 * 1e3:.3f} ms (+ drain {(t2 - t1) * 1e3:.2f} ms) | "
              f"read 4 devs {(t3 - t2) / 20 * 1e3:.3f} ms", flush=True)
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
