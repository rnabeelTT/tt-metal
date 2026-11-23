# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused block attention residual mix (tt/kernels/mixw_*.cpp) (the decode's one tile row, prefill chunks):

    lg = concat(sources) @ Qrep              (ttnn matmul; Qrep [H, nq*32]: query q in all 32 columns of tile q)
    w_iq = softmax_i(lg_iq)                  (kernel, fp32 SFPU)
    out_q = sum_i w_iq * S_i                 (kernel)

Replaces the max / sub / exp / sum / recip / mul and the per-query slice / mul / sum of Ops.mix (3 ops per mix
instead of ~11). Cores take (H column block, row tiles); each core computes its row tiles' softmax once."""

from __future__ import annotations

import hashlib

import ttnn

from .bconv import KDIR, _cb, _range_set, _rt

F32 = 4096
MAX_N = 6  # DST: max + n weights + sum <= 8 fp32 tiles


ROWS = __import__("os").environ.get("I2M_MIX_ROWS", "1") == "1"


def _rows_def(T):
    """<= 16 live rows (decode): the compute's SFPU work on those rows only (mixw_compute.cpp MIX_ROWS)."""
    return [("MIX_ROWS", str(T))] if ROWS and T <= 16 else []


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
        self.G = grid.x * grid.y
        self.grid_x = grid.x
        self.splits = {}

    def _split(self, RT):
        """Row tiles x H column blocks over the cores: nb column blocks (a divisor of Ht, nb * RT <= cores when
        possible), each block's row tiles in contiguous ranges -> (cores, per, [(t0, r0, nr)]). One softmax per row
        tile per core (the T <= 32 decode: RT = 1, the H tiles over the cores as before)."""
        if RT not in self.splits:
            nb = max(d for d in range(1, self.Ht + 1) if self.Ht % d == 0 and (d * RT <= self.G or d == 1))
            cpb = self.G // nb  # cores per column block
            per = self.Ht // nb
            work = []
            for b in range(nb):
                k = min(cpb, RT)
                for j in range(k):
                    r0, r1 = RT * j // k, RT * (j + 1) // k
                    work.append((b * per, r0, r1 - r0))
            cores = [ttnn.CoreCoord(i % self.grid_x, i // self.grid_x) for i in range(len(work))]
            self.splits[RT] = (cores, per, work)
        return self.splits[RT]

    def __call__(self, S, lg, n, nq, out_mc=ttnn.L1_MEMORY_CONFIG):
        """S [1, n, T, H] fp32 interleaved, lg [1, n, T, nq*32] fp32 interleaved -> nq fp32 [1, 1, T, H]
        (T <= 32, or a multiple of 32: one softmax per row tile)."""
        T, H = int(S.shape[2]), int(S.shape[3])
        assert n <= MAX_N and (T <= 32 or T % 32 == 0)
        RT = max(1, T // 32)
        outs = [ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, H]), ttnn.float32, ttnn.TILE_LAYOUT, self.mesh,
                                               out_mc) for _ in range(nq)]
        cores, per, work = self._split(RT)
        crs = _range_set(cores)
        cb_lg, cb_s, cb_w, cb_out = 0, 1, 2, 3
        f32 = ttnn.float32
        nbuf = 2 if RT > 1 else 1  # double-buffer the row tiles' logits
        cbs = [_cb(crs, cb_lg, f32, F32, nbuf * n * nq), _cb(crs, cb_s, f32, F32, n * min(per, 4)),
               _cb(crs, cb_w, f32, F32, n * nq), _cb(crs, cb_out, f32, F32, 2)]
        rd_ct = [cb_lg, cb_s, n, nq, self.Ht, RT] + ttnn.TensorAccessorArgs(S).get_compile_time_args() \
            + ttnn.TensorAccessorArgs(lg).get_compile_time_args()
        rd_rt = [[S.buffer_address(), lg.buffer_address(), t0, per, r0, nr] for t0, r0, nr in work]
        wr_ct = [cb_out, nq] + ttnn.TensorAccessorArgs(outs[0]).get_compile_time_args()
        wr_rt = [[t0, per, r0, nr, self.Ht] + [o.buffer_address() for o in outs] for t0, r0, nr in work]
        u2d = [ttnn.UnpackToDestMode.Default] * 64
        for i in (cb_lg, cb_s, cb_w):
            u2d[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           dst_full_sync_en=True, math_approx_mode=False)
        cfg.unpack_to_dest_mode = u2d
        dm = lambda src, ct, rt, riscv, noc: ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=str(KDIR / src), core_ranges=crs, compile_time_args=ct, runtime_args=_rt(cores, rt),
            config=ttnn.DataMovementConfigDescriptor(processor=riscv, noc=noc))
        kernels = [
            dm("mixw_reader.cpp", rd_ct, rd_rt, ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
            dm("mixw_writer.cpp", wr_ct, wr_rt, ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "mixw_compute.cpp"), core_ranges=crs,
                                  compile_time_args=[cb_lg, cb_s, cb_w, cb_out, n, nq],
                                  runtime_args=_rt(cores, [[per, nr] for _, _, nr in work]), config=cfg,
                                  defines=_rows_def(T)),
        ]
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        key = ("i2m_mixw2", n, nq, T, H, len(cores), repr(S.memory_config()), repr(lg.memory_config()), repr(out_mc))
        prog.custom_program_hash = int.from_bytes(hashlib.sha1(repr(key).encode()).digest()[:8], "little") & ((1 << 63) - 1)
        ttnn.generic_op([S, lg] + outs, prog)
        return outs

    def multi(self, srcs, lgs, nq, out_mc=ttnn.L1_MEMORY_CONFIG):
        """One row tile (T <= 32) from separate tensors: srcs n fp32 [1, 1, T, H], lgs n (logits tensor, first tile)
        (each source's nq query tiles at tiles off .. off + nq - 1 of its tensor) -> nq fp32 [1, 1, T, H].
        (all srcs share one memory config, all logit tensors another)"""
        n = len(srcs)
        T, H = int(srcs[0].shape[2]), int(srcs[0].shape[3])
        assert n <= MAX_N and T <= 32 and len(lgs) == n
        outs = [ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, H]), ttnn.float32, ttnn.TILE_LAYOUT, self.mesh,
                                               out_mc) for _ in range(nq)]
        cb_lg, cb_s, cb_w, cb_out = 0, 1, 2, 3
        f32 = ttnn.float32
        cbs = [_cb(self.crs, cb_lg, f32, F32, n * nq), _cb(self.crs, cb_s, f32, F32, n * self.per),
               _cb(self.crs, cb_w, f32, F32, n * nq), _cb(self.crs, cb_out, f32, F32, 2)]
        nc = len(self.cores)
        rd_ct = [cb_lg, cb_s, n, nq] + ttnn.TensorAccessorArgs(srcs[0]).get_compile_time_args() \
            + ttnn.TensorAccessorArgs(lgs[0][0]).get_compile_time_args()
        per_src = [v for s_, (lg, off) in zip(srcs, lgs) for v in (s_.buffer_address(), lg.buffer_address(), int(off))]
        rd_rt = [[i * self.per, self.per] + per_src for i in range(nc)]
        wr_ct = [cb_out, nq] + ttnn.TensorAccessorArgs(outs[0]).get_compile_time_args()
        wr_rt = [[i * self.per, self.per, 0, 1, self.Ht] + [o.buffer_address() for o in outs] for i in range(nc)]
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
            dm("mixm_reader.cpp", rd_ct, rd_rt, ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
            dm("mixw_writer.cpp", wr_ct, wr_rt, ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "mixw_compute.cpp"), core_ranges=self.crs,
                                  compile_time_args=[cb_lg, cb_s, cb_w, cb_out, n, nq],
                                  runtime_args=_rt(self.cores, [[self.per, 1]] * nc), config=cfg,
                                  defines=_rows_def(T)),
        ]
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        key = ("i2m_mixm2", n, nq, T, H, nc, repr(srcs[0].memory_config()), repr(lgs[0][0].memory_config()), repr(out_mc))
        prog.custom_program_hash = int.from_bytes(hashlib.sha1(repr(key).encode()).digest()[:8], "little") & ((1 << 63) - 1)
        ttnn.generic_op(list(srcs) + [lg for lg, _ in lgs] + outs, prog)
        return outs

