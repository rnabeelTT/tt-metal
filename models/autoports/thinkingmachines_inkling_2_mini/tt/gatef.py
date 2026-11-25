# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill message-gate tail with its forget / gain matmul in one program (tt/kernels/gatef_*.cpp):

    out = h - tanh(g @ Wf) * h + tanh(g @ Wg) * m          g [T, R] bf16, W = [Wf | Wg] [R, 2H] bf16, h / m fp32

Replaces the fp32 [T, 2H] pre-tanh matmul (written to DRAM and read back) and the tail program (tt/gatet.py).
Bitwise equal to them: the same products (g and W are bf16 values either way), the R K-tiles accumulated in DST.
Cores take (column group, row range); each keeps its columns' weight tiles."""

from __future__ import annotations

import hashlib

import ttnn

from .bconv import KDIR, _cb, _range_set, _rt

F32, BF = 4096, 2048


class GateFused:
    def __init__(self, mesh, H: int, R: int):
        self.mesh, self.Ht, self.R = mesh, H // 32, R // 32
        grid = mesh.compute_with_storage_grid_size()
        self.G, self.gx = grid.x * grid.y, grid.x
        self.cg = 4 if self.Ht % 4 == 0 else 1  # columns per core
        self.splits = {}
        self.progs = {}

    def _split(self, RT):
        if RT not in self.splits:
            ncg = self.Ht // self.cg
            rs = max(1, min(RT, self.G // ncg))
            work = [(c * self.cg, self.cg, RT * i // rs, RT * (i + 1) // rs - RT * i // rs)
                    for c in range(ncg) for i in range(rs)]
            cores = [ttnn.CoreCoord(i % self.gx, i // self.gx) for i in range(len(work))]
            self.splits[RT] = (cores, work)
        return self.splits[RT]

    def __call__(self, g, w, h, m, out_mc=ttnn.DRAM_MEMORY_CONFIG):
        """g [1, 1, T, R] bf16, w [1, 1, R, 2H] bf16, h / m [1, 1, T, H] fp32 (T % 32 == 0) -> [1, 1, T, H] fp32."""
        T = int(h.shape[-2])
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, 32 * self.Ht]), ttnn.float32, ttnn.TILE_LAYOUT,
                                             self.mesh, out_mc)
        key = (T, h.memory_config().buffer_type, m.memory_config().buffer_type, out_mc.buffer_type)
        if key not in self.progs:
            assert T % 32 == 0 and g.dtype == ttnn.bfloat16 and w.dtype == ttnn.bfloat16
            assert h.dtype == ttnn.float32 and m.dtype == ttnn.float32
            self.progs[key] = self._build(T, g, w, h, m, out)
        rd, wr, cm, cbs, hh = self.progs[key]
        rd.common_runtime_args = [g.buffer_address(), w.buffer_address(), h.buffer_address(), m.buffer_address()]
        wr.common_runtime_args = [out.buffer_address()]
        prog = ttnn.ProgramDescriptor(kernels=[rd, wr, cm], semaphores=[], cbs=cbs)
        prog.custom_program_hash = hh
        ttnn.generic_op([g, w, h, m, out], prog)
        return out

    def _build(self, T, g, w, h, m, out):
        cores, work = self._split(T // 32)
        crs = _range_set(cores)
        R = self.R
        cb_g, cb_w, cb_hm, cb_out = 0, 1, 2, 16
        bf, f32 = ttnn.bfloat16, ttnn.float32
        cbs = [_cb(crs, cb_g, bf, BF, 2 * R), _cb(crs, cb_w, bf, BF, 2 * R * self.cg),
               _cb(crs, cb_hm, f32, F32, 4 * self.cg), _cb(crs, cb_out, f32, F32, 2)]
        rd_ct = [cb_g, cb_w, cb_hm, self.Ht, R]
        for t in (g, w, h, m):
            rd_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        wr_ct = [cb_out, self.Ht] + ttnn.TensorAccessorArgs(out).get_compile_time_args()
        per_core = [[c0, nc, r0, nr] for c0, nc, r0, nr in work]
        u2d = [ttnn.UnpackToDestMode.Default] * 64
        u2d[cb_hm] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           dst_full_sync_en=False, math_approx_mode=False)
        cfg.unpack_to_dest_mode = u2d
        dm = lambda src, ct, riscv, noc: ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=str(KDIR / src), core_ranges=crs, compile_time_args=ct, runtime_args=_rt(cores, per_core),
            config=ttnn.DataMovementConfigDescriptor(processor=riscv, noc=noc))
        rd = dm("gatef_reader.cpp", rd_ct, ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0)
        wr = dm("gatef_writer.cpp", wr_ct, ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1)
        cm = ttnn.KernelDescriptor(kernel_source=str(KDIR / "gatef_compute.cpp"), core_ranges=crs,
                                   compile_time_args=[cb_g, cb_w, cb_hm, cb_out, R],
                                   runtime_args=_rt(cores, [[nc, nr] for _, nc, _, nr in work]), config=cfg)
        key = ("i2m_gatef3", self.Ht, R, T, len(cores), repr(g.memory_config()), repr(h.memory_config()),
               repr(m.memory_config()), repr(out.memory_config()))
        hh = int.from_bytes(hashlib.sha1(repr(key).encode()).digest()[:8], "little") & ((1 << 63) - 1)
        return rd, wr, cm, cbs, hh
