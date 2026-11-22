# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused block attention residual mix (tt/kernels/mixw_*.cpp) for one tile row (T <= 32, the decode):

    lg = concat(sources) @ Qrep              (ttnn matmul; Qrep [H, nq*32]: query q in all 32 columns of tile q)
    w_iq = softmax_i(lg_iq)                  (kernel, fp32 SFPU)
    out_q = sum_i w_iq * S_i                 (kernel)

Replaces the max / sub / exp / sum / recip / mul and the per-query slice / mul / sum of Ops.mix (3 ops per mix
instead of ~11). H tiles split over the cores; every core recomputes the (tiny) softmax."""

from __future__ import annotations

import hashlib

import ttnn

from .bconv import KDIR, _cb, _range_set, _rt

F32 = 4096
MAX_N = 6  # DST: max + n weights + sum <= 8 fp32 tiles


class MixKernel:
    def __init__(self, mesh, H: int):
        self.mesh, self.Ht = mesh, H // 32
        grid = mesh.compute_with_storage_grid_size()
        nc = min(self.Ht, grid.x * grid.y)
        while self.Ht % nc:
            nc -= 1
        self.cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(nc)]
        self.per = self.Ht // nc
        self.crs = _range_set(self.cores)

    def __call__(self, S, lg, n, nq, out_mc=ttnn.L1_MEMORY_CONFIG):
        """S [1, n, T, H] fp32 interleaved, lg [1, n, T, nq*32] fp32 interleaved -> nq fp32 [1, 1, T, H]."""
        assert n <= MAX_N and int(S.shape[2]) <= 32
        T, H = int(S.shape[2]), int(S.shape[3])
        outs = [ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, H]), ttnn.float32, ttnn.TILE_LAYOUT, self.mesh,
                                               out_mc) for _ in range(nq)]
        cb_lg, cb_s, cb_w, cb_out = 0, 1, 2, 3
        f32 = ttnn.float32
        cbs = [_cb(self.crs, cb_lg, f32, F32, n * nq), _cb(self.crs, cb_s, f32, F32, n * self.per),
               _cb(self.crs, cb_w, f32, F32, n), _cb(self.crs, cb_out, f32, F32, 2)]
        nc = len(self.cores)
        rd_ct = [cb_lg, cb_s, n, nq, self.Ht] + ttnn.TensorAccessorArgs(S).get_compile_time_args() \
            + ttnn.TensorAccessorArgs(lg).get_compile_time_args()
        rd_rt = [[S.buffer_address(), lg.buffer_address(), i * self.per, self.per] for i in range(nc)]
        wr_ct = [cb_out, nq] + ttnn.TensorAccessorArgs(outs[0]).get_compile_time_args()
        wr_rt = [[i * self.per, self.per] + [o.buffer_address() for o in outs] for i in range(nc)]
        u2d = [ttnn.UnpackToDestMode.Default] * 64
        for i in (cb_lg, cb_s, cb_w):
            u2d[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           dst_full_sync_en=True, math_approx_mode=False)
        cfg.unpack_to_dest_mode = u2d
        dm = lambda src, ct, rt, riscv, noc: ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=str(KDIR / src), core_ranges=self.crs, compile_time_args=ct, runtime_args=_rt(self.cores, rt),
            config=ttnn.DataMovementConfigDescriptor(processor=riscv, noc=noc))
        kernels = [
            dm("mixw_reader.cpp", rd_ct, rd_rt, ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
            dm("mixw_writer.cpp", wr_ct, wr_rt, ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "mixw_compute.cpp"), core_ranges=self.crs,
                                  compile_time_args=[cb_lg, cb_s, cb_w, cb_out, n, nq],
                                  runtime_args=_rt(self.cores, [[self.per]] * nc), config=cfg),
        ]
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        key = ("i2m_mixw", n, nq, T, H, nc, repr(S.memory_config()), repr(lg.memory_config()), repr(out_mc))
        prog.custom_program_hash = int.from_bytes(hashlib.sha1(repr(key).encode()).digest()[:8], "little") & ((1 << 63) - 1)
        ttnn.generic_op([S, lg] + outs, prog)
        return outs
