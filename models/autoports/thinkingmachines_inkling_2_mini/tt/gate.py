# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Message gate in ONE program (tt/kernels/gate_*.cpp), one tile row (T <= 32, the decode):

    out = h - tanh(Wf g) h + tanh(Wg g) m,   g = bf16(Wd [bf16(rms(h)) | bf16(rms(m))])

C cores own H / 32 / C hidden tiles each; two cross-core rounds through a root core (the sums of squares, the rank-R
partial down projections), everything else local. Replaces ~13 ops (two sharded norms with their reshards, typecasts,
concat, two matmuls on 2 cores, the tail kernel)."""

from __future__ import annotations

import hashlib
import struct

import ttnn

from .bconv import KDIR, _cb, _range_set, _rt

F32, BF = 4096, 2048
ROWS = __import__("os").environ.get("I2M_GATE_ROWS", "1") == "1"  # SFPU passes on the live rows only (<= 16 rows)


def _bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


class GateKernel:
    """Shared by every gate (one core set); the per-gate weights are passed per call."""

    def __init__(self, mesh, H: int, R: int, eps: float, C: int = 24):
        self.mesh, self.Ht, self.R, self.eps = mesh, H // 32, R // 32, eps
        assert self.Ht % C == 0 and self.R == 2, (H, R, C)
        self.C, self.PER = C, self.Ht // C
        grid = mesh.compute_with_storage_grid_size()
        self.cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(C)]
        self.crs = _range_set(self.cores)
        self.noc = [mesh.worker_core_from_logical_core(c) for c in self.cores]

    def __call__(self, h, m, wd, wfg, out_mc=ttnn.L1_MEMORY_CONFIG):
        """h, m [1, 1, T, H] fp32 (interleaved), wd [1, 1, 2H, R] bf16, wfg [1, 1, R, 2H] bf16 -> [1, 1, T, H] fp32."""
        h, m = (ttnn.sharded_to_interleaved(t, ttnn.L1_MEMORY_CONFIG) if t.is_sharded() else t for t in (h, m))
        assert h.dtype == ttnn.float32 and m.dtype == ttnn.float32 and int(h.shape[-2]) <= 32
        out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(h.shape)), ttnn.float32, ttnn.TILE_LAYOUT, self.mesh, out_mc)
        C, PER, R = self.C, self.PER, self.R
        (cb_h, cb_m, cb_wd, cb_wf, cb_wg, cb_ones, cb_sq, cb_part1, cb_gath, cb_res1, cb_r, cb_hn, cb_part2, cb_res2,
         cb_g, cb_fa, cb_out) = range(17)
        f32, bf = ttnn.float32, ttnn.bfloat16
        crs = self.crs
        cbs = [_cb(crs, cb_h, f32, F32, PER), _cb(crs, cb_m, f32, F32, PER), _cb(crs, cb_wd, bf, BF, 2 * PER * R),
               _cb(crs, cb_wf, bf, BF, R * PER), _cb(crs, cb_wg, bf, BF, R * PER), _cb(crs, cb_ones, bf, BF, 1),
               _cb(crs, cb_sq, f32, F32, 4 * PER), _cb(crs, cb_part1, f32, F32, 2), _cb(crs, cb_gath, f32, F32, 2 * C),
               _cb(crs, cb_res1, f32, F32, 2), _cb(crs, cb_r, f32, F32, 2), _cb(crs, cb_hn, bf, BF, 2 * PER),
               _cb(crs, cb_part2, f32, F32, R), _cb(crs, cb_res2, bf, BF, R), _cb(crs, cb_g, bf, BF, R),
               _cb(crs, cb_fa, f32, F32, 2 * PER), _cb(crs, cb_out, f32, F32, 2)]
        rd_ct = [cb_h, cb_m, cb_wd, cb_wf, cb_wg, cb_ones, PER, R, self.Ht] \
            + ttnn.TensorAccessorArgs(h).get_compile_time_args() + ttnn.TensorAccessorArgs(wd).get_compile_time_args() \
            + ttnn.TensorAccessorArgs(wfg).get_compile_time_args()
        rd_rt = [[h.buffer_address(), m.buffer_address(), wd.buffer_address(), wfg.buffer_address(), i * PER]
                 for i in range(C)]
        cp_ct = [cb_h, cb_m, cb_wd, cb_wf, cb_wg, cb_ones, cb_sq, cb_part1, cb_gath, cb_res1, cb_r, cb_hn, cb_part2,
                 cb_res2, cb_g, cb_fa, cb_out, C, PER, R, _bits(1.0 / (self.Ht * 32)), _bits(self.eps)]
        cp_rt = [[i] for i in range(C)]
        wr_ct = [cb_part1, cb_gath, cb_res1, cb_r, cb_part2, cb_res2, cb_g, cb_out, C, PER, R] \
            + ttnn.TensorAccessorArgs(out).get_compile_time_args()
        all_xy = [v for p in self.noc for v in (p.x, p.y)]
        root = self.noc[0]
        wr_rt = [[out.buffer_address(), i * PER, i, root.x, root.y] + (all_xy if i == 0 else []) for i in range(C)]
        u2d = [ttnn.UnpackToDestMode.Default] * 64
        for i in (cb_h, cb_m, cb_gath, cb_r, cb_fa):
            u2d[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           dst_full_sync_en=True, math_approx_mode=False)
        cfg.unpack_to_dest_mode = u2d
        dm = lambda src, ct, rt, riscv, noc: ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=str(KDIR / src), core_ranges=crs, compile_time_args=ct, runtime_args=_rt(self.cores, rt),
            config=ttnn.DataMovementConfigDescriptor(processor=riscv, noc=noc))
        kernels = [
            dm("gate_reader.cpp", rd_ct, rd_rt, ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
            dm("gate_writer.cpp", wr_ct, wr_rt, ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "gate_compute.cpp"), core_ranges=crs, compile_time_args=cp_ct,
                                  runtime_args=_rt(self.cores, cp_rt), config=cfg,
                                  defines=[("GATE_ROWS", str(int(h.shape[-2])))] + ([] if ROWS else [("GATE_FULL", "1")])),
        ]
        sems = [ttnn.SemaphoreDescriptor(i, ttnn.CoreType.WORKER, crs, 0) for i in range(6)]
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        key = ("i2m_gate", C, PER, R, int(h.shape[-2]), repr(h.memory_config()), repr(m.memory_config()),
               repr(wd.memory_config()), repr(out_mc))
        prog.custom_program_hash = int.from_bytes(hashlib.sha1(repr(key).encode()).digest()[:8], "little") & ((1 << 63) - 1)
        ttnn.generic_op([h, m, wd, wfg, out], prog)
        return out
