# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Microbenchmark: the host step's small replicated row-major bf16 uploads (crafted logits rows [1, 1, 4, 224], host
partial [1, 1, 4, 3072]) through pin.PinnedRows.put vs copy_host_to_device_tensor; idle device, and each put right
after a small readback (the host step's order); then the in-loop suspects for the pinned put: an idle gap before each put
(the host waiting on the device), a busy device (matmuls queued ahead), and DRAM-streaming threads (the CPU experts).
usage (in the image): python bench_put.py"""

import sys
import threading
import time

import numpy as np

sys.path.insert(0, "/poc")
import torch  # noqa: E402

import ttnn  # noqa: E402
from hybrid.pin import Pinx, PinnedRows  # noqa: E402

N = 500


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576, trace_region_size=10_000_000)
    rep = ttnn.ReplicateTensorToMesh(mesh)
    px = Pinx(mesh)
    rows = PinnedRows(px)
    rd = ttnn.from_torch(
        torch.zeros(1, 1, 4, 288), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep
    )
    for W in (224, 3072):
        dst = ttnn.from_torch(
            torch.zeros(1, 1, 4, W),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=rep,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        t = torch.randn(1, 1, 4, W).to(torch.bfloat16)
        rows.put(0, dst, t)
        ttnn.synchronize_device(mesh)
        res = {}
        for mode in ("pinned", "pinned+read", "c2d", "c2d+read"):
            ttnn.synchronize_device(mesh)
            rows.t_copy = rows.t_write = 0.0
            tot = 0.0
            for _ in range(N):
                if mode.endswith("+read"):
                    ttnn.to_torch(ttnn.get_device_tensors(rd)[0])
                t0 = time.perf_counter()
                if mode.startswith("pinned"):
                    rows.put(0, dst, t)
                else:
                    ttnn.copy_host_to_device_tensor(ttnn.from_torch(t, dtype=ttnn.bfloat16, mesh_mapper=rep), dst)
                tot += time.perf_counter() - t0
            ttnn.synchronize_device(mesh)
            res[mode] = tot / N * 1e6
            if mode.startswith("pinned"):
                res[mode + " (write call)"] = rows.t_write / N * 1e6
        print(f"W={W}: " + "  ".join(f"{k} {v:.1f} us" for k, v in res.items()), flush=True)
        suspects(mesh, rows, dst, t, W)
        back = ttnn.to_torch(ttnn.get_device_tensors(dst)[0])
        print(f"  readback matches: {bool((back.float() == t.float()).all())}", flush=True)
    px.release()
    ttnn.close_mesh_device(mesh)


def suspects(mesh, rows, dst, t, W):
    a = ttnn.from_torch(torch.randn(1, 1, 2048, 4096), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    b = ttnn.from_torch(torch.randn(1, 1, 4096, 4096), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    big = np.ones(256 << 20, dtype=np.float32)  # 1 GB, streamed by the load threads
    stop = threading.Event()

    def stream():
        while not stop.is_set():
            big.sum()

    def timed(pre=None):
        ttnn.synchronize_device(mesh)
        tot = 0.0
        for _ in range(200):
            if pre:
                pre()
            t0 = time.perf_counter()
            rows.put(0, dst, t)
            tot += time.perf_counter() - t0
        ttnn.synchronize_device(mesh)
        return tot / 200 * 1e6

    def spin(us):
        t_end = time.perf_counter() + us * 1e-6
        while time.perf_counter() < t_end:
            pass

    res = {"tight": timed()}
    res["sleep 300us"] = timed(lambda: time.sleep(300e-6))
    res["spin 300us"] = timed(lambda: spin(300))
    res["device busy"] = timed(lambda: ttnn.deallocate(ttnn.matmul(a, b)))
    th = [threading.Thread(target=stream, daemon=True) for _ in range(4)]
    for x in th:
        x.start()
    time.sleep(0.2)
    res["4 DRAM threads"] = timed()
    res["4 DRAM threads + sleep 300us"] = timed(lambda: time.sleep(300e-6))
    stop.set()
    for x in th:
        x.join()
    print(f"  W={W} put: " + "  ".join(f"{k} {v:.1f} us" for k, v in res.items()), flush=True)


if __name__ == "__main__":
    main()
