# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Indexer pooling tail (decode, tt/indexer.Hy4Indexer._pool_fused after the value / gate matmul) with the ttnn
chain's numerics (HY4_IDX_POOL_K): the elementwise ops around the one real reduction folded into 2 generic_ops, the
reduction (ttnn.rms_norm of the pooled keys) kept as the ttnn call, the rope + concat as RopeFusedOp.tail (same
values as slice + rope + concat):

    pool  g_j = vg_j[:, d:] + ape_j; m = max_j g_j; e_j = exp(g_j - m); pooled = (sum_j e_j v_j) / (sum_j e_j)
          (bias add, slices, ttnn.max / ttnn.sum over the r = 4 slots, subtract, exp, multiply, div: ~12 programs)
          n    = ttnn.rms_norm(pooled)
    post  keys = bf16(n * kn)                                        (multiply, typecast)
          out  = rope.tail(keys, keep=True)                          (slice, slice, rope, concat)

Each fused op uses the same SFPU function and operand order as the ttnn op it replaces (IEEE add / mul / max, the same
exp_tile / div_binary_tile, slot sums in ttnn.sum(dim=1)'s order) -> bit-identical keys (tests/test_idx_pool_exact.py).
The earlier one-program kernel (tt/idx_pool: row sums as FP32 hi / lo matmuls, its own softmax order) moved the long3k
decode gate (.8273/.2930 -> .8236/.3042, journal:hy4-fuse:0016)."""

from __future__ import annotations

from pathlib import Path

import ttnn

KDIR = Path(__file__).parent / "kernels"
TILE = 32


class IdxPoolExactOp:
    _OPS = {}

    @classmethod
    def get(cls, mesh):
        if id(mesh) not in cls._OPS:
            cls._OPS[id(mesh)] = cls(mesh)
        return cls._OPS[id(mesh)]

    def __init__(self, mesh, max_cores=32):
        self.mesh, self.max_cores = mesh, max_cores

    def _run(self, mode, a, b, out, NT, compute, fp32_out):
        nunits = 4 * NT
        nc = min(nunits, self.max_cores)
        per = [nunits // nc + (1 if i < nunits % nc else 0) for i in range(nc)]
        starts = [sum(per[:i]) for i in range(nc)]
        cores = [ttnn.CoreCoord(i % 8, i // 8) for i in range(nc)]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        acc = lambda t: ttnn.TensorAccessorArgs(t).get_compile_time_args()  # noqa: E731
        F, B = ttnn.float32, ttnn.bfloat16
        out_page = 4096 if fp32_out else 2048

        def cb(i, dt, n):
            page = 4096 if dt == F else 2048
            fmt = ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)
            return ttnn.CBDescriptor(total_size=page * n, core_ranges=crs, format_descriptors=[fmt])

        nin = 8 if mode == 0 else 1
        cbs = [cb(0, F, 2 * nin), cb(1, F, 2 * nin), cb(16, F if fp32_out else B, 2)]

        def rt(args):
            r = ttnn.RuntimeArgs()
            for c, x in zip(cores, args):
                r[c.x][c.y] = x
            return r

        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           math_approx_mode=False, dst_full_sync_en=True)
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[0] = modes[1] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = modes
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "px_reader.cpp"), core_ranges=crs,
                                  compile_time_args=[mode, NT] + acc(a) + acc(b),
                                  runtime_args=rt([[a.buffer_address(), b.buffer_address(), s, n]
                                                   for s, n in zip(starts, per)]),
                                  config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1)),
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "px_writer.cpp"), core_ranges=crs,
                                  compile_time_args=[out_page] + acc(out),
                                  runtime_args=rt([[out.buffer_address(), s, n] for s, n in zip(starts, per)]),
                                  config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0)),
            ttnn.KernelDescriptor(kernel_source=str(KDIR / compute), core_ranges=crs, compile_time_args=[0],
                                  runtime_args=rt([[n] for n in per]), config=cfg),
        ]
        ttnn.generic_op([a, b, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return out

    def pool(self, vg, bias, n, d, mem):
        """vg [1, 1, 4 n, 2d] fp32 (slot-major rows), bias [1, 4, n, 2d] fp32 -> pooled [1, 1, n, d] fp32."""
        assert d == 128 and n % TILE == 0
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, n, d]), ttnn.float32, ttnn.TILE_LAYOUT, self.mesh, mem)
        return self._run(0, vg, bias, out, n // TILE, "px_pool_compute.cpp", True)

    def post(self, nrm, kn, mem):
        """nrm [1, 1, n, d] fp32, kn [1, 1, 1, d] fp32 -> bf16(nrm * kn) [1, 1, n, d]."""
        n, d = int(nrm.shape[-2]), int(nrm.shape[-1])
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, n, d]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, mem)
        return self._run(1, nrm, kn, out, n // TILE, "px_post_compute.cpp", False)
