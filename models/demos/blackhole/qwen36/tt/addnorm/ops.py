# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode residual add + RMSNorm in one ``ttnn.generic_op`` program on the replicated-residual cores
(decode_boundary.py: [1, 1, B, H] width-sharded over NC = 16 cores, T = H / NC / 32 tiles each):

    h   = residual + reduced            (``reduced`` = the boundary all-reduce result; skipped when None)
    out = h * rsqrt(mean(h^2) + eps) * gamma

Each core squares its T tiles and sums them per row (matmul with an all-ones tile), sends that tile to every core
(NC x 1 or 2 KB: faces 0 / 2, then a semaphore increment each), sums the NC tiles and normalizes its shard. residual,
reduced and h are CBs on their own shards (no copies). ``h`` comes back in the
residual layout, ``out`` L1 interleaved (what the mixer / MoE matmuls read; replaces the sharded norm + the
sharded->interleaved copy). gamma (``prepare``): [1, 1, 32, H] bf16, every row = gamma (RMSNorm.weight, unit offset
included).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.gdn_decode.ops import TILE, _bits, _cb, _ndev, _per_device, _range_set, _rt

KERNEL_DIR = Path(__file__).parent / "kernels"
SEM = 0


class AddNorm:
    def __init__(self, mesh_device, hidden, cores=(8, 2), eps=1e-6):
        from models.demos.blackhole.qwen36.tt.moe_decode.experts import noc_coords

        self.mesh = mesh_device
        self.H = hidden
        self.cores = [ttnn.CoreCoord(i % cores[0], i // cores[0]) for i in range(cores[0] * cores[1])]  # shard order
        self.NC = len(self.cores)
        assert hidden % (self.NC * TILE) == 0
        self.T = hidden // self.NC // TILE
        self.eps = eps
        self.crs = _range_set(self.cores)
        coords = noc_coords(mesh_device)
        self.noc = [v for c in self.cores for v in coords[(c.x, c.y)]]
        self._rep = {"mesh_mapper": ttnn.ReplicateTensorToMesh(mesh_device)} if _ndev(mesh_device) > 1 else {}
        self.ones = ttnn.from_torch(torch.ones(1, 1, TILE, TILE), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                    device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG, **self._rep)

    def prepare(self, gamma):
        """gamma: torch vector of H or a replicated ttnn tensor (e.g. RMSNorm.weight) -> [1, 1, 32, H] bf16, every row."""
        g = gamma if isinstance(gamma, torch.Tensor) else _per_device(gamma)[0]
        g = g.reshape(-1)[: self.H].float()
        return ttnn.from_torch(g.reshape(1, 1, 1, self.H).expand(1, 1, TILE, self.H).contiguous(), dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                               **self._rep)

    def __call__(self, residual, reduced, gamma):
        """residual / reduced: [1, 1, B, H] bf16 in the residual layout (reduced may be None). Returns (h, out): h the
        new residual (None when reduced is None), out [1, 1, B, H] bf16 L1 interleaved."""
        add = reduced is not None
        shape = residual.shape
        out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh,
                                             ttnn.L1_MEMORY_CONFIG)
        h = (ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh,
                                            residual.memory_config()) if add else None)
        T, NC = self.T, self.NC
        bf, f32 = (ttnn.bfloat16, 2048), (ttnn.float32, 4096)
        # residual / reduced / h: CBs on the tensors' own shards (no copies)
        cbs = [ttnn.cb_descriptor_from_sharded_tensor(0, residual)]
        cbs += [ttnn.cb_descriptor_from_sharded_tensor(1, reduced), ttnn.cb_descriptor_from_sharded_tensor(4, h)] if add else []
        sizes = ([] if add else [(4, bf, 1)]) + [(2, bf, T), (3, bf, 1), (5, f32, T), (6, f32, 1), (7, f32, NC), (8, f32, 1), (9, bf, T)]
        cbs += [_cb(self.crs, i, dt, tb, n) for i, (dt, tb), n in sizes]
        faces = 1 if shape[-2] <= 16 else 2  # rows 0..15 live in face 0 (column 0 holds the row sums)
        rd_ct = [T, int(add), 0, 1, 2, 3] + ttnn.TensorAccessorArgs(gamma).get_compile_time_args() \
            + ttnn.TensorAccessorArgs(self.ones).get_compile_time_args()
        rd_rt = [[gamma.buffer_address(), self.ones.buffer_address(), i] for i in range(NC)]
        wr_ct = [T, NC, SEM, 6, 7, 9, faces] + ttnn.TensorAccessorArgs(out).get_compile_time_args()
        wr_rt = [[out.buffer_address(), i] + self.noc for i in range(NC)]
        fid = getattr(ttnn.MathFidelity, os.environ.get("QWEN35_ADDNORM_FIDELITY", "HiFi4"))
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=fid, fp32_dest_acc_en=True,
                                           math_approx_mode=False)
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "reader.cpp"), core_ranges=self.crs, compile_time_args=rd_ct,
                runtime_args=_rt(self.cores, rd_rt),
                config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "writer.cpp"), core_ranges=self.crs, compile_time_args=wr_ct,
                runtime_args=_rt(self.cores, wr_rt),
                config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "compute.cpp"), core_ranges=self.crs,
                compile_time_args=[T, int(add), NC, _bits(self.eps), _bits(1.0 / self.H)],
                runtime_args=_rt(self.cores, [[] for _ in range(NC)]), config=cfg,
            ),
        ]
        sem = ttnn.SemaphoreDescriptor(id=SEM, core_type=ttnn.CoreType.WORKER, core_ranges=self.crs, initial_value=0)
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[sem], cbs=cbs)
        io = [residual] + ([reduced] if add else []) + [gamma, self.ones] + ([h] if add else []) + [out]
        ttnn.generic_op(io, program)
        return h, out
