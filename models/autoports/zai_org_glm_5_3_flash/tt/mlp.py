# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Dense SwiGLU MLP of GLM-5.3-Flash layers 0-2 (intermediate 12288) with the HF swiglu_limit clamp:

    gate = min(x @ Wg^T, L);  up = clamp(x @ Wu^T, -L, L);  out = (silu(gate) * up) @ Wd^T      (L = 10)

On a TP mesh the intermediate dim is sharded (column-parallel gate/up, row-parallel down); the block returns
per-device partial sums and the caller owns the reduction.
"""

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

DRAM = ttnn.DRAM_MEMORY_CONFIG


class DenseMLP(LightweightModule):
    def __init__(
        self,
        mesh_device,
        gate: torch.Tensor,
        up: torch.Tensor,
        down: torch.Tensor,
        swiglu_limit: float,
        *,
        dtype=ttnn.bfloat8_b,
        cache_prefix=None,
    ):
        self.limit = float(swiglu_limit)
        tp = tuple(mesh_device.shape)[1]
        col = ttnn.ShardTensorToMesh(mesh_device, dim=-1) if tp > 1 else ttnn.ReplicateTensorToMesh(mesh_device)
        row = ttnn.ShardTensorToMesh(mesh_device, dim=-2) if tp > 1 else ttnn.ReplicateTensorToMesh(mesh_device)

        def up_(t, mapper, name):
            return ttnn.as_tensor(
                t.contiguous(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=DRAM,
                mesh_mapper=mapper,
                cache_file_name=f"{cache_prefix}.{name}" if cache_prefix else None,
            )

        self.wg = up_(gate.T, col, "gate")
        self.wu = up_(up.T, col, "up")
        self.wd = up_(down.T, row, "down")
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        g = ttnn.linear(x, self.wg, compute_kernel_config=self.ckc, memory_config=DRAM)
        g2 = ttnn.silu(ttnn.minimum(g, self.limit), memory_config=DRAM)
        ttnn.deallocate(g)
        u = ttnn.linear(x, self.wu, compute_kernel_config=self.ckc, memory_config=DRAM)
        u2 = ttnn.clamp(u, min=-self.limit, max=self.limit, memory_config=DRAM)
        ttnn.deallocate(u)
        h = ttnn.multiply(g2, u2, memory_config=DRAM)
        ttnn.deallocate(g2)
        ttnn.deallocate(u2)
        out = ttnn.linear(h, self.wd, compute_kernel_config=self.ckc, memory_config=DRAM)
        ttnn.deallocate(h)
        return out
