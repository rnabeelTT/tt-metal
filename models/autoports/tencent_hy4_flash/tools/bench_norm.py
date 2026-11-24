# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark: decode RMSNorm [1, 1, 32, W] interleaved (one core per tile row = 1 core) vs width-sharded
multi-core (tt/common.RMSNorm decode path); exactness and device time from trace replays."""

from __future__ import annotations

import torch
import ttnn

from models.autoports.tencent_hy4_flash.tools.bench_dsh import timed
from models.autoports.tencent_hy4_flash.tt.common import RMSNorm


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=50_000_000)
    try:
        rep = ttnn.ReplicateTensorToMesh(mesh)
        for W in (3072, 1024, 512, 128):
            torch.manual_seed(0)
            x = torch.randn(1, 1, 32, W) * 3
            g = torch.randn(W) * 0.2 + 1
            xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep,
                                 memory_config=ttnn.L1_MEMORY_CONFIG)
            n = RMSNorm(mesh, g, 1e-5)
            t_i, oi = timed(mesh, lambda: ttnn.rms_norm(xt, epsilon=n.eps, weight=n.w, compute_kernel_config=n.ckc,
                                                        memory_config=ttnn.L1_MEMORY_CONFIG))
            t_s, os_ = timed(mesh, lambda: n(xt, ttnn.L1_MEMORY_CONFIG))
            a = ttnn.to_torch(ttnn.get_device_tensors(oi)[0]).float()
            b = ttnn.to_torch(ttnn.get_device_tensors(os_)[0]).float()
            print(f"W {W}: interleaved {t_i:6.1f} us | decode path {t_s:6.1f} us | equal {torch.equal(a, b)} "
                  f"max diff {(a - b).abs().max():.3e}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
