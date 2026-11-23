# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The sorted prefill MoE's activation in one program (tt/kernels/moeact_*.cpp): the grouped gate|up output
[1, G, gs, 2W] bf16 and the dispatched rows' routing weights [1, 1, Rd, 1] bf16 -> the down projection's input
[1, G, gs, W] bf16 = bf16(bf16(tanh(g / 4) sigmoid(g) tanh(u)) w). Replaces the slices, typecasts, the activation's
ops, the weight multiply and the reshapes (tt/moe.py inkling_act_fused + the weights)."""

from __future__ import annotations

import hashlib
import struct

import ttnn

from .bconv import KDIR, _cb, _range_set, _rt


class MoEAct:
    def __init__(self, mesh):
        self.mesh = mesh
        g = mesh.compute_with_storage_grid_size()
        self.gx, self.ncores = g.x, g.x * g.y

    def __call__(self, gu, weights, out):
        G, gs, W2 = int(gu.shape[1]), int(gu.shape[2]), int(gu.shape[3])
        Wt, Rt = W2 // 64, G * gs // 32
        n = min(self.ncores, Rt)
        bounds = [Rt * i // n for i in range(n + 1)]
        cores = [ttnn.CoreCoord(i % self.gx, i // self.gx) for i in range(n)]
        crs = _range_set(cores)
        cb_g, cb_u, cb_w, cb_out = 0, 1, 2, 3
        bf = ttnn.bfloat16
        cbs = [_cb(crs, cb_g, bf, 2048, 2), _cb(crs, cb_u, bf, 2048, 2), _cb(crs, cb_w, bf, 2048, 2),
               _cb(crs, cb_out, bf, 2048, 2)]
        rt = [[bounds[i], bounds[i + 1]] for i in range(n)]
        dm = ttnn.DataMovementConfigDescriptor
        rd = ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "moeact_reader.cpp"), core_ranges=crs,
            compile_time_args=[cb_g, cb_u, cb_w, Wt] + ttnn.TensorAccessorArgs(gu).get_compile_time_args()
            + ttnn.TensorAccessorArgs(weights).get_compile_time_args(),
            runtime_args=_rt(cores, rt), config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0))
        rd.common_runtime_args = [gu.buffer_address(), weights.buffer_address()]
        wr = ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "moeact_writer.cpp"), core_ranges=crs,
            compile_time_args=[cb_out, Wt] + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
            runtime_args=_rt(cores, rt), config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1))
        wr.common_runtime_args = [out.buffer_address()]
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           dst_full_sync_en=True, math_approx_mode=False)
        quarter = struct.unpack("<I", struct.pack("<f", 0.25))[0]
        cp = ttnn.KernelDescriptor(kernel_source=str(KDIR / "moeact_compute.cpp"), core_ranges=crs,
                                   compile_time_args=[cb_g, cb_u, cb_w, cb_out, Wt, quarter],
                                   runtime_args=_rt(cores, [[r[1] - r[0]] for r in rt]), config=cfg)
        prog = ttnn.ProgramDescriptor(kernels=[rd, wr, cp], semaphores=[], cbs=cbs)
        key = ("i2m_moeact", G, gs, W2, n, tuple(repr(t.memory_config()) for t in (gu, weights, out)))
        prog.custom_program_hash = int.from_bytes(hashlib.sha1(repr(key).encode()).digest()[:8], "little") & ((1 << 63) - 1)
        ttnn.generic_op([gu, weights, out], prog)
        return out
