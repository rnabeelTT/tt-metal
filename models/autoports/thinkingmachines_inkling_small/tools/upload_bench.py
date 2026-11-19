# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host time of a small host -> device upload while the device is busy (as in a prefill, where each layer uploads
routing ids / scores / host-expert rows behind queued device work): ttnn.from_torch onto the mesh (allocates, writes)
vs ttnn.copy_host_to_device_tensor into a persistent buffer. A blocking path shows the queued device time in its
host time.

usage (in the image): tools/run_dev.sh python models/autoports/thinkingmachines_inkling_small/tools/upload_bench.py
"""

import time

import torch

import ttnn


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=64_000_000)
    rep = ttnn.ReplicateTensorToMesh(mesh)
    try:
        a = ttnn.from_torch(torch.randn(1, 1, 2048, 4096), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                            mesh_mapper=rep)
        w = ttnn.from_torch(torch.randn(1, 1, 4096, 4096), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh,
                            mesh_mapper=rep)

        def busy(n):  # ~n matmuls queued on CQ0
            for _ in range(n):
                ttnn.deallocate(ttnn.matmul(a, w))

        busy(2)
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        busy(20)
        ttnn.synchronize_device(mesh)
        t_busy = time.perf_counter() - t0
        print(f"20 queued matmuls: {t_busy * 1e3:.1f} ms of device work", flush=True)

        cases = {
            "ids [1,1,2048,8] int32 tile": (torch.randint(0, 190, (1, 1, 2048, 8), dtype=torch.int32), ttnn.uint32, ttnn.TILE_LAYOUT),
            "idx [1,2048] int32 row-major": (torch.randint(0, 100, (1, 2048), dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            "table [1024,4096] bf16 row-major": (torch.randn(1024, 4096).to(torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        }
        for name, (t, dt, lay) in cases.items():
            persist = ttnn.from_torch(t, dtype=dt, layout=lay, device=mesh, mesh_mapper=rep, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for label in ("from_torch (alloc + write)", "host tensor + copy_host_to_device_tensor"):
                res = []
                for _ in range(3):
                    ttnn.synchronize_device(mesh)
                    busy(20)  # device busy for ~t_busy
                    t1 = time.perf_counter()
                    if label.startswith("from_torch"):
                        d = ttnn.from_torch(t, dtype=dt, layout=lay, device=mesh, mesh_mapper=rep,
                                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
                        res.append(time.perf_counter() - t1)
                        ttnn.deallocate(d)
                    else:
                        h = ttnn.from_torch(t, dtype=dt, layout=lay, mesh_mapper=rep)
                        t2 = time.perf_counter()
                        ttnn.copy_host_to_device_tensor(h, persist)
                        res.append(time.perf_counter() - t1)
                        conv = t2 - t1
                ttnn.synchronize_device(mesh)
                extra = f" (host conversion {conv * 1e3:.2f} ms)" if not label.startswith("from_torch") else ""
                print(f"{name:34s} {label:44s} host {min(res) * 1e3:7.2f} - {max(res) * 1e3:7.2f} ms{extra}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
