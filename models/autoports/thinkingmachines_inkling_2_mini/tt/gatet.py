# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused message-gate tail (tt/kernels/gatet_*.cpp) for one tile row (T <= 32, the decode):

    out = h - tanh(f) * h + tanh(a) * m          fg = [f | a] (pre-tanh, [1, 1, T, 2H] fp32), h, m fp32

Replaces tanh / 2 slices / mul / sub / mul / add (7 ops) of MessageGate with one program."""

from __future__ import annotations

import hashlib

import ttnn

from .bconv import KDIR, _cb, _range_set, _rt

F32 = 4096


class GateTail:
    def __init__(self, mesh, H: int):
        self.mesh, self.Ht = mesh, H // 32
        grid = mesh.compute_with_storage_grid_size()
        nc = min(self.Ht, grid.x * grid.y)
        while self.Ht % nc:
            nc -= 1
        self.cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(nc)]
        self.per = self.Ht // nc
        self.crs = _range_set(self.cores)

    def __call__(self, fg, h, m, out_mc=ttnn.L1_MEMORY_CONFIG):
        ins = [ttnn.sharded_to_interleaved(t, ttnn.L1_MEMORY_CONFIG) if t.is_sharded() else t for t in (fg, h, m)]
        fg, h, m = ins
        assert all(t.dtype == ttnn.float32 for t in ins) and int(h.shape[-2]) <= 32
        out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(h.shape)), ttnn.float32, ttnn.TILE_LAYOUT, self.mesh, out_mc)
        cb_in, cb_out = 0, 1
        cbs = [_cb(self.crs, cb_in, ttnn.float32, F32, 8), _cb(self.crs, cb_out, ttnn.float32, F32, 2)]
        nc = len(self.cores)
        rd_ct = [cb_in, self.Ht] + ttnn.TensorAccessorArgs(fg).get_compile_time_args() \
            + ttnn.TensorAccessorArgs(h).get_compile_time_args() + ttnn.TensorAccessorArgs(m).get_compile_time_args()
        rd_rt = [[fg.buffer_address(), h.buffer_address(), m.buffer_address(), i * self.per, self.per] for i in range(nc)]
        wr_ct = [cb_out] + ttnn.TensorAccessorArgs(out).get_compile_time_args()
        wr_rt = [[out.buffer_address(), i * self.per, self.per] for i in range(nc)]
        u2d = [ttnn.UnpackToDestMode.Default] * 64
        u2d[cb_in] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           dst_full_sync_en=True, math_approx_mode=False)
        cfg.unpack_to_dest_mode = u2d
        dm = lambda src, ct, rt, riscv, noc: ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=str(KDIR / src), core_ranges=self.crs, compile_time_args=ct, runtime_args=_rt(self.cores, rt),
            config=ttnn.DataMovementConfigDescriptor(processor=riscv, noc=noc))
        kernels = [
            dm("gatet_reader.cpp", rd_ct, rd_rt, ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
            dm("gatet_writer.cpp", wr_ct, wr_rt, ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "gatet_compute.cpp"), core_ranges=self.crs,
                                  compile_time_args=[cb_in, cb_out], runtime_args=_rt(self.cores, [[self.per]] * nc),
                                  config=cfg),
        ]
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        key = ("i2m_gatet", self.Ht, nc, int(h.shape[-2]), repr(fg.memory_config()), repr(h.memory_config()),
               repr(m.memory_config()), repr(out_mc))
        prog.custom_program_hash = int.from_bytes(hashlib.sha1(repr(key).encode()).digest()[:8], "little") & ((1 << 63) - 1)
        ttnn.generic_op([fg, h, m, out], prog)
        return out
