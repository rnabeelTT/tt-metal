# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Hybrid-MoE POC, microbenchmark 3: host->device writes while the CPU expert kernel streams DDR.

Loops 1.75 MB/die (one 397B bfp4 expert, sharded over 4 dies) writes on CQ1 for argv[1] seconds and prints GB/s
per second. Run bench2_cpu.py <seconds> alongside it in a second process to measure both under contention.
"""

import json
import sys
import time

import torch
import ttnn


def main():
    secs = float(sys.argv[1])
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), num_command_queues=2)
    try:
        sh = ttnn.ShardTensorToMesh(mesh, dim=-1)
        wcols = int(1.75e6 / 0.5625 / 4096 // 32 * 32)
        src = torch.randn(1, 1, 4096, wcols * 4, dtype=torch.bfloat16)
        # 64 distinct host buffers (~450 MB) so the copies stream from DDR, not from the 32 MB L3
        pool = [
            ttnn.from_torch(
                src + i, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=sh
            )
            for i in range(64)
        ]
        dev_t = ttnn.from_torch(src, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=sh)
        nbytes = 4096 * wcols * 4 * 0.5625
        print("READY", flush=True)
        end = time.perf_counter() + secs
        while time.perf_counter() < end:
            t0, n = time.perf_counter(), 0
            while time.perf_counter() - t0 < 1.0:
                ttnn.copy_host_to_device_tensor(pool[n % len(pool)], dev_t, cq_id=1)
                n += 1
            ttnn.synchronize_device(mesh)
            dt = time.perf_counter() - t0
            print(json.dumps({"t": round(time.time(), 1), "h2d_GB_per_s": round(n * nbytes / dt / 1e9, 2)}), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
