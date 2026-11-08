# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Microbenchmark: the host step's small replicated row-major bf16 uploads (crafted logits rows [1, 1, 4, 224], host
partial [1, 1, 4, 3072]) through pin.PinnedRows.put vs copy_host_to_device_tensor; idle device, and each put right
after a small readback (the host step's order). usage (in the image): python bench_put.py"""

import sys
import time

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
        back = ttnn.to_torch(ttnn.get_device_tensors(dst)[0])
        print(f"  readback matches: {bool((back.float() == t.float()).all())}", flush=True)
    px.release()
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
