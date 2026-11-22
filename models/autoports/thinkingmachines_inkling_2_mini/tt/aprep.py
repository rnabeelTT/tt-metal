# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode attention prep in one program (tt/kernels/aprep_*.cpp), B <= 32 users, one row each:

    pq [1, 1, B, q | r | sink | gate] fp32, k / v [1, 1, B, NKV * HD] bf16 (conv outputs)  ->
    q  [1, B, NH, HD]  bf16   l2-clipped (qk_clip) (* tau_b on global layers)
    r  [1, B, NH, 32]  bf16   l2-clipped (rel_clip) (* tau_b), DR (16) columns + zero padding (the SDPA's rq)
    k  [1, B, 32, HD]  bf16   l2-clipped, padded to 32 head rows, in the KV-write layout (height-sharded)
    v  [1, B, 32, HD]  bf16   unweighted RMS
    sk [1, B, NH, 32]  bf16   sink / scale in every column

Replaces ~35 ttnn ops per application (slices, reshapes, typecasts, the clip / rms chains, tau, pads, the sink
broadcast, the KV reshards). Units (user, tensor) are spread over the cores."""

from __future__ import annotations

import hashlib
import struct

import ttnn

from .bconv import KDIR, _cb, _range_set, _rt

F32, BF = 4096, 2048


def _bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


class AttnPrep:
    def __init__(self, w, B: int, kv_mc):
        assert B <= 32 and w.DR in (16, 32) and w.NH < 32
        self.w, self.B, self.kv_mc, self.mesh = w, B, kv_mc, w.mesh
        grid = self.mesh.compute_with_storage_grid_size()
        self.total = 5 * B
        nc = min(self.total, grid.x * grid.y)
        self.cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(nc)]
        self.crs = _range_set(self.cores)
        self.HT = w.HD // 32

    def __call__(self, pq, k, v, tau=None):
        w, B, HT, mesh = self.w, self.B, self.HT, self.mesh
        L1 = ttnn.L1_MEMORY_CONFIG
        k = k if (not k.is_sharded() and k.memory_config().buffer_type == ttnn.BufferType.L1) else \
            ttnn.to_memory_config(k, L1)
        v = v if (not v.is_sharded() and v.memory_config().buffer_type == ttnn.BufferType.L1) else \
            ttnn.to_memory_config(v, L1)
        glob = not w.sliding
        al = lambda shape, mc: ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16,  # noqa: E731
                                                               ttnn.TILE_LAYOUT, mesh, mc)
        q_o = al([1, B, w.NH, w.HD], L1)
        r_o = al([1, B, w.NH, 32], L1)
        k_o = al([1, B, 32, w.HD], self.kv_mc)
        v_o = al([1, B, 32, w.HD], self.kv_mc)
        s_o = al([1, B, w.NH, 32], L1)
        tau_t = tau if glob else pq  # (unused on sliding layers)
        cb_xf, cb_xb, cb_tau, cb_ones, cb_sq, cb_f, cb_out = 0, 1, 2, 3, 4, 5, 6
        f32, bf = ttnn.float32, ttnn.bfloat16
        cbs = [_cb(self.crs, cb_xf, f32, F32, 2 * HT), _cb(self.crs, cb_xb, bf, BF, 2 * HT),
               _cb(self.crs, cb_tau, f32, F32, 2), _cb(self.crs, cb_ones, bf, BF, 1),
               _cb(self.crs, cb_sq, f32, F32, 2 * HT), _cb(self.crs, cb_f, f32, F32, 2),
               _cb(self.crs, cb_out, bf, BF, 2 * HT)]
        nc = len(self.cores)
        R0 = w.qd // 32
        S0 = (w.qd + w.rd) // 32
        rd_ct = [cb_xf, cb_xb, cb_tau, cb_ones, w.NH, w.NKV, HT, w.DR, R0, S0, int(glob)] \
            + ttnn.TensorAccessorArgs(pq).get_compile_time_args() + ttnn.TensorAccessorArgs(k).get_compile_time_args() \
            + ttnn.TensorAccessorArgs(tau_t).get_compile_time_args()
        rd_rt = [[pq.buffer_address(), k.buffer_address(), v.buffer_address(), tau_t.buffer_address(), i, nc,
                  self.total] for i in range(nc)]
        cp_ct = [cb_xf, cb_xb, cb_tau, cb_ones, cb_sq, cb_f, cb_out, HT, w.DR // 32, int(glob), _bits(w.qk_clip),
                 _bits(w.rel_clip), _bits(1.0 / w.HD), _bits(w.eps), _bits(1.0 / w.scale)]
        cp_rt = [[i, nc, self.total] for i in range(nc)]
        wr_ct = [cb_out, HT] + ttnn.TensorAccessorArgs(q_o).get_compile_time_args() \
            + ttnn.TensorAccessorArgs(r_o).get_compile_time_args() + ttnn.TensorAccessorArgs(k_o).get_compile_time_args()
        wr_rt = [[q_o.buffer_address(), r_o.buffer_address(), k_o.buffer_address(), v_o.buffer_address(),
                  s_o.buffer_address(), i, nc, self.total] for i in range(nc)]
        u2d = [ttnn.UnpackToDestMode.Default] * 64
        for i in (cb_xf, cb_f):
            u2d[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           dst_full_sync_en=True, math_approx_mode=False)
        cfg.unpack_to_dest_mode = u2d
        dm = lambda src, ct, rt, riscv, noc: ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=str(KDIR / src), core_ranges=self.crs, compile_time_args=ct, runtime_args=_rt(self.cores, rt),
            config=ttnn.DataMovementConfigDescriptor(processor=riscv, noc=noc))
        kernels = [
            dm("aprep_reader.cpp", rd_ct, rd_rt, ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
            dm("aprep_writer.cpp", wr_ct, wr_rt, ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "aprep_compute.cpp"), core_ranges=self.crs,
                                  compile_time_args=cp_ct, runtime_args=_rt(self.cores, cp_rt), config=cfg),
        ]
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        key = ("i2m_aprep", B, nc, w.NH, w.NKV, w.HD, glob, w.qk_clip, w.rel_clip, repr(pq.memory_config()),
               repr(k.memory_config()), repr(tau_t.memory_config()), repr(self.kv_mc), int(pq.shape[-1]))
        prog.custom_program_hash = int.from_bytes(hashlib.sha1(repr(key).encode()).digest()[:8], "little") & ((1 << 63) - 1)
        ttnn.generic_op([pq, k, v, tau_t, q_o, r_o, k_o, v_o, s_o], prog)
        return q_o, r_o, k_o, v_o, s_o
