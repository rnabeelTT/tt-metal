# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused router post-processing of the decode MoE (tt/moe.py router_row): one generic_op instead of add + sigmoid + add
+ concat + slice. a, b: the hi / lo halves of the streamed router's FP32 logits [1, 1, 32, E]; bias_rows: the
selection bias [1, 1, 32, E] FP32 (row-replicated) -> row [1, 1, T, 2E] FP32 = [s + bias | s], s = sigmoid(a + b).
The same SFPU arithmetic as the ttnn ops (FP32 add, sigmoid_tile<RC, accurate>, FP32 add). One output column tile of
each half per core. kernels/rp_*.cpp."""

from __future__ import annotations

from pathlib import Path

import ttnn

KDIR = Path(__file__).parent / "kernels"
TILE = 32


class RouterPostOp:
    _OPS = {}

    @classmethod
    def get(cls, mesh):
        if id(mesh) not in cls._OPS:
            cls._OPS[id(mesh)] = cls(mesh)
        return cls._OPS[id(mesh)]

    def __init__(self, mesh):
        self.mesh = mesh
        g = mesh.compute_with_storage_grid_size()
        self.grid = [ttnn.CoreCoord(x, y) for y in range(g.y) for x in range(g.x)]

    def __call__(self, a, b, bias_rows, T, mem=None):
        E = int(a.shape[-1])
        n = E // TILE
        nc = min(n, 16)
        per = -(-n // nc)
        nc = -(-n // per)
        cores = self.grid[:nc]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        y = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, 2 * E]), ttnn.float32, ttnn.TILE_LAYOUT, self.mesh,
                                           mem or ttnn.L1_MEMORY_CONFIG)

        def cb(i, k):
            fmt = ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=4096)
            return ttnn.CBDescriptor(total_size=4096 * k, core_ranges=crs, format_descriptors=[fmt])

        cbs = [cb(0, 2), cb(1, 2), cb(2, 2), cb(16, 2), cb(17, 2)]
        acc = lambda t: ttnn.TensorAccessorArgs(t).get_compile_time_args()  # noqa: E731
        rng = [(k * per, min(n, (k + 1) * per)) for k in range(nc)]

        def rt(args):
            r = ttnn.RuntimeArgs()
            for c, x in zip(cores, args):
                r[c.x][c.y] = x
            return r

        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           math_approx_mode=False)
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[0] = modes[1] = modes[2] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = modes
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "rp_reader.cpp"), core_ranges=crs,
                                  compile_time_args=acc(a) + acc(b) + acc(bias_rows),
                                  runtime_args=rt([[a.buffer_address(), b.buffer_address(), bias_rows.buffer_address(),
                                                    s, e] for s, e in rng]),
                                  config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1)),
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "rp_writer.cpp"), core_ranges=crs,
                                  compile_time_args=[n] + acc(y), runtime_args=rt([[y.buffer_address(), s, e]
                                                                                   for s, e in rng]),
                                  config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0)),
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "rp_compute.cpp"), core_ranges=crs,
                                  compile_time_args=[], runtime_args=rt([[e - s] for s, e in rng]), config=cfg),
        ]
        ttnn.generic_op([a, b, bias_rows, y], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return y
