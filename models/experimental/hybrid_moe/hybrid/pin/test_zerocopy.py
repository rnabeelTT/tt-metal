# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Do pinx writes land? Expert-sized writes (random pinned-region offsets -> random slots of a bfp4 DRAM tensor), read
back with pinx_read_at and compared byte for byte, per die. Zero-copy (pinned region read over the NOC) on CQ0 / CQ1 and
the copy path as the control. With device work queued between the writes when ZC_BUSY=1.

usage (inside the image, e.g. tools/run_dev.sh python3 models/experimental/hybrid_moe/hybrid/pin/test_zerocopy.py [n])
"""

import os
import sys

import numpy as np
import torch

import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from hybrid.pin import Pinx  # noqa: E402


def main():
    n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    busy = os.environ.get("ZC_BUSY", "1") == "1"
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), num_command_queues=2)
    try:
        R, C = 8192, 16384  # per device: 131072 bfp4 tiles, 75 MB
        t = ttnn.from_torch(
            torch.zeros(1, 1, R, C),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        a = ttnn.from_torch(
            torch.randn(1, 1, 1024, 2048),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        b = ttnn.from_torch(
            torch.randn(1, 1, 2048, 2048),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        px = Pinx(mesh)
        dst, size = px.dst(t)
        tp = px.tp
        n = 576 * 7744  # ~4.46 MB: one expert's per-die bytes
        slots = size // n
        src_rid, src = px.region(512 << 20)
        src[:] = np.random.default_rng(0).integers(0, 256, src.size, dtype=np.uint8)
        back_rid, back = px.region(tp * n)
        rng = np.random.default_rng(1)
        for mode, cq in (("copy", 0), ("zerocopy", 0), ("zerocopy", 1), ("copy", 1)):
            px.zerocopy = mode == "zerocopy"
            bad = np.zeros(tp, dtype=np.int64)
            for it in range(n_iter):
                j = int(rng.integers(0, slots))
                offs = [int(rng.integers(0, (len(src) - n) // 4096)) * 4096 for _ in range(tp)]
                if cq == 1:
                    ttnn.wait_for_event(1, ttnn.record_event(mesh, cq_id=0))
                px.write_at(dst, src_rid, offs, j * n, n, cq_id=cq)
                if busy:  # device work behind the write on CQ0
                    if cq == 1:
                        ttnn.wait_for_event(0, ttnn.record_event(mesh, cq_id=1))
                    for _ in range(4):
                        ttnn.deallocate(ttnn.matmul(a, b))
                ttnn.synchronize_device(mesh)
                px.read_at(dst, back_rid, [d * n for d in range(tp)], j * n, n, cq_id=0)
                for d in range(tp):
                    if not np.array_equal(back[d * n : (d + 1) * n], src[offs[d] : offs[d] + n]):
                        bad[d] += 1
            print(f"{mode} CQ{cq}: {n_iter} writes of {n} bytes, mismatches per die {bad.tolist()}", flush=True)
    finally:
        os._exit(0)  # (close_mesh_device can hang here; the container is removed by the caller)


if __name__ == "__main__":
    main()
