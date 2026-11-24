# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused RoPE: one generic_op instead of tt/rope.DeviceRope's 6 ttnn ops (typecast, rotate-half matmul, 2 multiplies,
add, typecast). Same arithmetic: x (bf16) and its rotate-half (an exact FPU product with the +-1 permutation tile) in
FP32 DST, y = x * cos + rh * sin on the SFPU in FP32, narrowed to bf16 with ttnn.typecast's RNE. x [b, h, T, 64] bf16
TILE, cos / sin [1, 1, T, 64] fp32 TILE (broadcast over b, h). The interleaved pairs never cross a 32-column tile, so
P is one 32 x 32 tile. kernels/rope_*.cpp."""

from __future__ import annotations

from pathlib import Path

import ttnn

KDIR = Path(__file__).parent / "kernels"
TILE = 32


class RopeFusedOp:
    _OPS = {}

    @classmethod
    def get(cls, mesh):
        if id(mesh) not in cls._OPS:
            cls._OPS[id(mesh)] = cls(mesh)
        return cls._OPS[id(mesh)]

    def __init__(self, mesh):
        from models.autoports.tencent_hy4_flash.tt.rope import rotate_half_matrix

        self.mesh = mesh
        g = mesh.compute_with_storage_grid_size()
        self.grid = [ttnn.CoreCoord(x, y) for y in range(g.y) for x in range(g.x)]
        kw = {"mesh_mapper": ttnn.ReplicateTensorToMesh(mesh)} if mesh.get_num_devices() > 1 else {}
        self.P = ttnn.from_torch(rotate_half_matrix(TILE).reshape(1, 1, TILE, TILE), dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw)
        self.max_cores = 32

    @staticmethod
    def supports(x, cos, sin) -> bool:
        s = tuple(int(v) for v in x.shape)
        return (len(s) == 4 and s[-1] == 64 and s[-2] % TILE == 0 and x.dtype == ttnn.bfloat16
                and x.layout == ttnn.TILE_LAYOUT and cos.dtype == ttnn.float32 and sin.dtype == ttnn.float32
                and tuple(int(v) for v in cos.shape) == (1, 1, s[-2], 64)
                and tuple(int(v) for v in sin.shape) == (1, 1, s[-2], 64))

    @staticmethod
    def supports_tail(x, cos, sin) -> bool:
        """tail(): x [b, h, T, W] bf16 TILE, W a multiple of 32 >= 64, cos / sin [1, 1, T, 64] fp32."""
        s = tuple(int(v) for v in x.shape)
        return (len(s) == 4 and s[-1] % TILE == 0 and s[-1] >= 64 and s[-2] % TILE == 0 and x.dtype == ttnn.bfloat16
                and x.layout == ttnn.TILE_LAYOUT and cos.dtype == ttnn.float32 and sin.dtype == ttnn.float32
                and tuple(int(v) for v in cos.shape) == (1, 1, s[-2], 64)
                and tuple(int(v) for v in sin.shape) == (1, 1, s[-2], 64))

    def tail(self, x, cos, sin, mem=None, keep=False):
        """RoPE of x's last 64 columns read in place (no slice program): keep=False -> [b, h, T, 64]; keep=True ->
        [b, h, T, W], the first W - 64 columns copied (replaces slice + slice + rope + concat). Same values."""
        return self.__call__(x, cos, sin, mem, keep=keep, tail=True)

    def __call__(self, x, cos, sin, mem=None, keep=False, tail=False):
        s = tuple(int(v) for v in x.shape)
        TR = s[-2] // TILE
        WT = s[-1] // TILE if tail else 2
        n = s[0] * s[1] * TR * 2
        nc = min(n, self.max_cores)
        per = -(-n // nc)
        nc = -(-n // per)
        cores = self.grid[:nc]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        ys = list(s) if (keep or not tail) else s[:-1] + (64,)
        y = ttnn.allocate_tensor_on_device(ttnn.Shape(list(ys)), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh,
                                           mem or ttnn.DRAM_MEMORY_CONFIG)

        def cb(i, dt, tb, k):
            fmt = ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=tb)
            return ttnn.CBDescriptor(total_size=tb * k, core_ranges=crs, format_descriptors=[fmt])

        bf, f32 = ttnn.bfloat16, ttnn.float32
        cbs = [cb(0, bf, 2048, 2), cb(1, bf, 2048, 1), cb(2, f32, 4096, 2), cb(3, f32, 4096, 2), cb(16, bf, 2048, 2)]
        if keep and WT > 2:
            cbs.append(cb(17, bf, 2048, 2))
        acc = lambda t: ttnn.TensorAccessorArgs(t).get_compile_time_args()  # noqa: E731
        rng = [(k * per, min(n, (k + 1) * per)) for k in range(nc)]

        def rt(args):
            r = ttnn.RuntimeArgs()
            for c, a in zip(cores, args):
                r[c.x][c.y] = a
            return r

        rd_rt = [[x.buffer_address(), cos.buffer_address(), sin.buffer_address(), self.P.buffer_address(), a, b]
                 for a, b in rng]
        wr_rt = [[y.buffer_address(), a, b, x.buffer_address()] for a, b in rng]
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           math_approx_mode=False)
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[2] = modes[3] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = modes
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "rope_reader.cpp"), core_ranges=crs,
                                  compile_time_args=[TR, WT] + acc(x) + acc(cos) + acc(sin) + acc(self.P),
                                  runtime_args=rt(rd_rt),
                                  config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1)),
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "rope_writer.cpp"), core_ranges=crs,
                                  compile_time_args=[WT, int(keep and WT > 2)] + acc(y) + acc(x),
                                  runtime_args=rt(wr_rt),
                                  config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0)),
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "rope_compute.cpp"), core_ranges=crs,
                                  compile_time_args=[], runtime_args=rt([[b - a] for a, b in rng]), config=cfg),
        ]
        ttnn.generic_op([x, cos, sin, self.P, y], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return y
