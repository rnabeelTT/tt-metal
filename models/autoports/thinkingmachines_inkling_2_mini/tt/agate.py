# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The decode attention output gate in one program (tt/kernels/agate_*.cpp), <= 16 rows: the SDPA output
[1, B, NH, HD] (L1) -> [1, 1, B, NH * HD] times sigmoid(the gate columns of the fp32 q-projection output), rounded to
bf16 like the typecast. Replaces the reshape, the typecasts, the slice, the sigmoid and the multiply."""

from __future__ import annotations

import hashlib

import ttnn

from .bconv import KDIR, _cb, _range_set, _rt
from .bconv import _ks
from .post import SFPU_ROWS


class AttnGate:
    def __init__(self, mesh, NH, HD):
        self.mesh, self.NH, self.HD = mesh, NH, HD
        self.DT = HD // 32
        cols = NH * self.DT  # output tile columns: J per core on a full CX x CY rectangle
        grid = mesh.compute_with_storage_grid_size()
        for J in range(1, cols + 1):
            if cols % J:
                continue
            n = cols // J
            fit = [d for d in range(1, min(n, grid.x) + 1) if n % d == 0 and n // d <= grid.y]
            if fit:
                self.J, self.n, self.CX = J, n, max(fit)
                break
        self.cores = [ttnn.CoreCoord(i % self.CX, i // self.CX) for i in range(self.n)]
        self.crs = _range_set(self.cores)

    def __call__(self, o, pq, g0, out_mc=ttnn.L1_MEMORY_CONFIG):
        """o [1, B, NH, HD] bf16 (L1 interleaved), pq [1, 1, B, *] fp32 with the gate at column g0 (tile-aligned)
        -> [1, 1, B, NH * HD] bf16."""
        B = int(o.shape[1])
        assert B <= 16 and g0 % 32 == 0 and o.memory_config().buffer_type == ttnn.BufferType.L1
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, B, self.NH * self.HD]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                             self.mesh, out_mc)
        cb_o, cb_g, cb_out = 0, 1, 2
        cbs = [_cb(self.crs, cb_o, ttnn.bfloat16, 2048, 2), _cb(self.crs, cb_g, ttnn.float32, 4096, 2),
               _cb(self.crs, cb_out, ttnn.bfloat16, 2048, 2)]
        defs = [("AG_ROWS", str(B)), ("AG_J", str(self.J)), ("SFPU_ROWS", str(int(SFPU_ROWS)))]
        rd = ttnn.KernelDescriptor(
            kernel_source=_ks(KDIR, "agate_reader.cpp"), core_ranges=self.crs,
            compile_time_args=[cb_o, cb_g, self.DT, g0 // 32, self.CX] + ttnn.TensorAccessorArgs(o).get_compile_time_args()
            + ttnn.TensorAccessorArgs(pq).get_compile_time_args(), defines=defs,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1))
        rd.common_runtime_args = [o.buffer_address(), pq.buffer_address()]
        wr = ttnn.KernelDescriptor(
            kernel_source=_ks(KDIR, "agate_writer.cpp"), core_ranges=self.crs,
            compile_time_args=[cb_out, self.CX] + ttnn.TensorAccessorArgs(out).get_compile_time_args(), defines=defs,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0))
        wr.common_runtime_args = [out.buffer_address()]
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           math_approx_mode=False)
        u2d = [ttnn.UnpackToDestMode.Default] * 64
        u2d[cb_g] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = u2d
        cp = ttnn.KernelDescriptor(kernel_source=_ks(KDIR, "agate_compute.cpp"), core_ranges=self.crs,
                                   compile_time_args=[cb_o, cb_g, cb_out], defines=defs, config=cfg)
        prog = ttnn.ProgramDescriptor(kernels=[rd, wr, cp], semaphores=[], cbs=cbs)
        key = ("i2m_agate", SFPU_ROWS, B, self.NH, self.HD, g0, self.J, repr(o.memory_config()), repr(pq.memory_config()), repr(out_mc))
        prog.custom_program_hash = int.from_bytes(hashlib.sha1(repr(key).encode()).digest()[:8], "little") & ((1 << 63) - 1)
        ttnn.generic_op([o, pq, out], prog)
        return out


class PrefillGate:
    """The packed prefill's attention output gate in one program (tt/kernels/pgate_*.cpp): o [1, 1, T, W] bf16 times
    sigmoid(pq's fp32 gate columns from g0), rounded to bf16 like the typecast: AttnGate's math on whole tiles. Replaces
    the gate slice, the typecasts, the sigmoid and the multiply."""

    def __init__(self, mesh):
        self.mesh = mesh
        g = mesh.compute_with_storage_grid_size()
        self.cores = [ttnn.CoreCoord(i % g.x, i // g.x) for i in range(g.x * g.y)]
        self._pc = {}  # shape key -> descriptors (the call sets the addresses)

    def __call__(self, o, pq, g0, out_mc=ttnn.DRAM_MEMORY_CONFIG):
        T, W, PQ = int(o.shape[2]), int(o.shape[3]), int(pq.shape[3])
        assert T % 32 == 0 and W % 32 == 0 and g0 % 32 == 0 and pq.dtype == ttnn.float32
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, W]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, out_mc)
        key = ("i2m_pgate", T, W, PQ, g0, repr(o.memory_config()), repr(pq.memory_config()), repr(out_mc))
        hit = self._pc.get(key)
        if hit is None:
            hit = self._pc[key] = self._desc(o, pq, out, T, W, PQ, g0, key)
        rd, wr, cp, cbs, h = hit
        rd.common_runtime_args = [o.buffer_address(), pq.buffer_address()]
        wr.common_runtime_args = [out.buffer_address()]
        prog = ttnn.ProgramDescriptor(kernels=[rd, wr, cp], semaphores=[], cbs=cbs)
        prog.custom_program_hash = h
        ttnn.generic_op([o, pq, out], prog)
        return out

    def _desc(self, o, pq, out, T, W, PQ, g0, key):
        n = (T // 32) * (W // 32)
        nc = min(len(self.cores), n)
        cores = self.cores[:nc]
        crs = _range_set(cores)
        span = [(n * i // nc, n * (i + 1) // nc) for i in range(nc)]
        cb_o, cb_g, cb_out = 0, 1, 2
        cbs = [_cb(crs, cb_o, ttnn.bfloat16, 2048, 2), _cb(crs, cb_g, ttnn.float32, 4096, 2),
               _cb(crs, cb_out, ttnn.bfloat16, 2048, 2)]
        rd = ttnn.KernelDescriptor(
            kernel_source=_ks(KDIR, "pgate_reader.cpp"), core_ranges=crs,
            compile_time_args=[cb_o, cb_g, W // 32, PQ // 32, g0 // 32] + ttnn.TensorAccessorArgs(o).get_compile_time_args()
            + ttnn.TensorAccessorArgs(pq).get_compile_time_args(), runtime_args=_rt(cores, [list(s) for s in span]),
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1))
        wr = ttnn.KernelDescriptor(
            kernel_source=_ks(KDIR, "pgate_writer.cpp"), core_ranges=crs,
            compile_time_args=[cb_out] + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
            runtime_args=_rt(cores, [list(s) for s in span]),
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0))
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           math_approx_mode=False)
        u2d = [ttnn.UnpackToDestMode.Default] * 64
        u2d[cb_g] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = u2d
        cp = ttnn.KernelDescriptor(kernel_source=_ks(KDIR, "pgate_compute.cpp"), core_ranges=crs,
                                   compile_time_args=[cb_o, cb_g, cb_out],
                                   runtime_args=_rt(cores, [[b - a] for a, b in span]), config=cfg)
        return rd, wr, cp, cbs, int.from_bytes(hashlib.sha1(repr(key).encode()).digest()[:8], "little") & ((1 << 63) - 1)
