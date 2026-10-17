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

from .weight_cache import cached

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
        cache=None,
    ):
        """gate / up / down: [out, in] tensors or thunks returning them (read only on a weight-cache miss);
        cache: the layer's tt/weight_cache.WeightCache (None = no cache)."""
        self.limit = float(swiglu_limit)
        tp = tuple(mesh_device.shape)[1]
        col = ttnn.ShardTensorToMesh(mesh_device, dim=-1) if tp > 1 else ttnn.ReplicateTensorToMesh(mesh_device)
        row = ttnn.ShardTensorToMesh(mesh_device, dim=-2) if tp > 1 else ttnn.ReplicateTensorToMesh(mesh_device)

        def up_(t, mapper, name):
            return cached(
                cache,
                f"mlp.{name}",
                lambda: ttnn.from_torch(
                    (t() if callable(t) else t).T.contiguous(),
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=DRAM,
                    mesh_mapper=mapper,
                ),
            )

        self.wg = up_(gate, col, "gate")
        self.wu = up_(up, col, "up")
        self.wd = up_(down, row, "down")
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            # HiFi4: HiFi2 truncates the activation mantissa -> the MLP output ~0.8 % low on real weights (stage review #2)
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        self.wgu = None  # packed [gate | up] per device (pack(); fused decoder)

    def pack(self):
        """Pack gate and up into one [hidden, 2 * I_local] weight per device (an on-device concat of whole tiles, so the
        bfp8 blocks are unchanged) and fold the clamps and SiLU into the multiply (fused decoder; bit-identical,
        tools/fusing_probe.py: 17.19 -> 8.18 ms at 2048 rows, 334 -> 338 us at 32 decode rows)."""
        self.wgu = ttnn.concat([self.wg, self.wu], dim=-1, memory_config=DRAM)
        ttnn.deallocate(self.wg)
        ttnn.deallocate(self.wu)
        self.wg = self.wu = None
        L = self.limit
        self.act_up = [ttnn.UnaryWithParam(ttnn.UnaryOpType.HARDTANH, -L, L)]
        self.act_gate = [ttnn.UnaryWithParam((ttnn.UnaryOpType.MINIMUM, L)), ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)]

    def _packed_hidden(self, x: ttnn.Tensor) -> ttnn.Tensor:
        gu = ttnn.linear(x, self.wgu, compute_kernel_config=self.ckc, memory_config=DRAM)
        shp = list(gu.shape)
        il = shp[-1] // 2
        g = ttnn.slice(gu, [0] * len(shp), shp[:-1] + [il], memory_config=DRAM)
        u = ttnn.slice(gu, [0] * (len(shp) - 1) + [il], shp, memory_config=DRAM)
        ttnn.deallocate(gu)
        h = ttnn.multiply(
            u, g, input_tensor_a_activations=self.act_up, input_tensor_b_activations=self.act_gate, memory_config=DRAM
        )
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        return h

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if self.wgu is not None:
            h = self._packed_hidden(x)
            out = ttnn.linear(h, self.wd, compute_kernel_config=self.ckc, memory_config=DRAM)
            ttnn.deallocate(h)
            return out
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
