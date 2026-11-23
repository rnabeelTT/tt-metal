# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused SwiGLU of the expert-sorted prefill MoE (tt/prefill_moe.py): one generic_op instead of slice x 2 + clamp x 2 +
multiply(silu) + reshape + multiply(routing weights). fused [1, G, 32, 2 Il] = [up | gate] per 32-row group, weights
[1, 1, Rd, 1] bf16 -> [1, 1, Rd, Il] bf16 = clamp(up, -L, L) * silu(clamp(gate, max=L)) * w, groups split over the grid.
Bit-identical to the ttnn chain: the same SFPU calls (clamp_tile, silu_tile, mul_binary_tile, unary_bcast COL) in the
same order on a 16-bit DEST as those ops (bf16 in / out, fp32_dest_acc_en off), their bf16 intermediates kept in DEST
(kernels/sw_compute.cpp). HY4_PF_SWIGLU (default 1) enables it (prefill_moe.py)."""

from __future__ import annotations

import struct
from pathlib import Path

import ttnn

KDIR = Path(__file__).parent / "kernels"
NEG_INF_BITS = 0xFF800000  # ttnn.clamp(max=L): min = float lowest, as bf16 (round to nearest even) = -inf


def _bits_bf16(v: float) -> int:
    """fp32 bits of v rounded to bf16 (round to nearest even), as ttnn passes a clamp bound for a bf16 input."""
    b = struct.unpack("<I", struct.pack("<f", float(v)))[0]
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return b


class SwigluPF:
    _OPS = {}
    B = 3  # tile columns per DEST pass (sw_compute.cpp)

    @classmethod
    def get(cls, mesh):
        if id(mesh) not in cls._OPS:
            cls._OPS[id(mesh)] = cls(mesh)
        return cls._OPS[id(mesh)]

    @classmethod
    def supports(cls, Il: int) -> bool:
        return Il % 32 == 0 and (Il // 32) % cls.B == 0

    def __init__(self, mesh):
        self.mesh = mesh
        g = mesh.compute_with_storage_grid_size()
        self.grid = [ttnn.CoreCoord(x, y) for y in range(g.y) for x in range(g.x)]

    def __call__(self, fused, weights, limit: float, mem=None):
        G, Il = int(fused.shape[1]), int(fused.shape[-1]) // 2
        NT = Il // 32
        Rd = G * 32
        nc = min(len(self.grid), G)
        base, rem = divmod(G, nc)
        rng, g0 = [], 0
        for k in range(nc):
            n = base + (1 if k < rem else 0)
            rng.append((g0, g0 + n))
            g0 += n
        cores = self.grid[:nc]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        y = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, Rd, Il]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh,
                                           mem or ttnn.DRAM_MEMORY_CONFIG)

        def cb(i, k):
            fmt = ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)
            return ttnn.CBDescriptor(total_size=2048 * k, core_ranges=crs, format_descriptors=[fmt])

        cbs = [cb(0, 4 * NT), cb(1, 2), cb(16, 2 * self.B)]
        acc = lambda t: ttnn.TensorAccessorArgs(t).get_compile_time_args()  # noqa: E731

        def rt(args):
            r = ttnn.RuntimeArgs()
            for c, x in zip(cores, args):
                r[c.x][c.y] = x
            return r

        # the ttnn ops' compute config: bf16 in / out -> fp32_dest_acc_en off, math_approx_mode off
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False,
                                           math_approx_mode=False)
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "sw_reader.cpp"), core_ranges=crs,
                                  compile_time_args=[NT] + acc(fused) + acc(weights),
                                  runtime_args=rt([[fused.buffer_address(), weights.buffer_address(), a, b]
                                                   for a, b in rng]),
                                  config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1)),
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "sw_writer.cpp"), core_ranges=crs,
                                  compile_time_args=[NT] + acc(y),
                                  runtime_args=rt([[y.buffer_address(), a, b] for a, b in rng]),
                                  config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0)),
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "sw_compute.cpp"), core_ranges=crs,
                                  compile_time_args=[NT, _bits_bf16(limit), _bits_bf16(-limit), NEG_INF_BITS],
                                  runtime_args=rt([[b - a] for a, b in rng]), config=cfg),
        ]
        ttnn.generic_op([fused, weights, y], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return y
