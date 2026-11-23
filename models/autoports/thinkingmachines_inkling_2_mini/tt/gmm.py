# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Grouped expert matmul for the sorted prefill MoE (tt/kernels/gmm_*.cpp): A [1, G, gs, K] bf16 (the dispatched
rows, groups sorted by expert), W [1, E, K, N] (this die's expert weights) and the groups' expert ids [1, G] uint16
-> C [1, G, gs, N] bf16. Core (b, s): output columns [b nb, (b + 1) nb) of a contiguous slice s of the groups; the
weight block is re-read only when the slice moves to the next expert. Same math as ttnn.sparse_matmul with the whole
K in one in0 block (the program's fidelity, FP32 DST, K accumulated in order); ttnn's 1D config puts the N / 32 output
tiles of gate|up (24 at TP4) on that many cores, this one spreads (column block, group slice) over the grid."""

from __future__ import annotations

import hashlib

import ttnn

from .bconv import KDIR, _cb, _range_set, _rt

TILE_BYTES = {ttnn.bfloat4_b: 576, ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}


class GroupedMM:
    def __init__(self, mesh, ckc_fidelity=ttnn.MathFidelity.HiFi2, max_cores=None):
        self.mesh, self.fid = mesh, ckc_fidelity
        g = mesh.compute_with_storage_grid_size()
        self.gx, self.ncores = g.x, max_cores or g.x * g.y

    def __call__(self, a, w, ids, out, E, nb=None):
        """a [1, G, gs, K] bf16, w [1, E, K, N] (bfp8 / bf16 tiles), ids [1, G] uint16 ROW_MAJOR, out [1, G, gs, N]."""
        G, gs, K = int(a.shape[1]), int(a.shape[2]), int(a.shape[3])
        N = int(w.shape[-1])
        Kt, Nt, mt = K // 32, N // 32, gs // 32
        w_page = TILE_BYTES[w.dtype]
        if nb is None:  # the column block: DST (8 FP32 tiles), Nt % nb == 0, the weight block <= ~512 KB of L1
            nb = max(d for d in range(1, 9) if Nt % d == 0 and Kt * d * w_page <= 512 * 1024)
        NB = Nt // nb
        S = max(1, min(G, self.ncores // NB))  # group slices
        bounds = [G * s // S for s in range(S + 1)]
        cores, rt = [], []
        for s in range(S):
            for b in range(NB):
                i = len(cores)
                cores.append(ttnn.CoreCoord(i % self.gx, i // self.gx))
                rt.append([b, bounds[s], bounds[s + 1]])
        crs = _range_set(cores)
        cb_a, cb_w, cb_ids, cb_out = 0, 1, 2, 3
        ids_bytes = (G * 2 + 63) // 64 * 64
        cbs = [_cb(crs, cb_a, ttnn.bfloat16, 2048, 2 * Kt), _cb(crs, cb_w, w.dtype, w_page, Kt * nb),
               _cb(crs, cb_ids, ttnn.uint16, ids_bytes, 1), _cb(crs, cb_out, ttnn.bfloat16, 2048, 2 * nb)]
        dm = ttnn.DataMovementConfigDescriptor
        rd = ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "gmm_reader.cpp"), core_ranges=crs,
            compile_time_args=[cb_a, cb_w, cb_ids, Kt, Nt, nb, mt, E, ids_bytes, w_page]
            + ttnn.TensorAccessorArgs(a).get_compile_time_args() + ttnn.TensorAccessorArgs(w).get_compile_time_args()
            + ttnn.TensorAccessorArgs(ids).get_compile_time_args(),
            runtime_args=_rt(cores, rt), config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0))
        rd.common_runtime_args = [a.buffer_address(), w.buffer_address(), ids.buffer_address()]
        wr = ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "gmm_writer.cpp"), core_ranges=crs,
            compile_time_args=[cb_out, Nt, nb, mt] + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
            runtime_args=_rt(cores, rt), config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1))
        wr.common_runtime_args = [out.buffer_address()]
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=self.fid, fp32_dest_acc_en=True, dst_full_sync_en=True,
                                           math_approx_mode=False)
        cp = ttnn.KernelDescriptor(kernel_source=str(KDIR / "gmm_compute.cpp"), core_ranges=crs,
                                   compile_time_args=[cb_a, cb_w, cb_out, Kt, nb, mt],
                                   runtime_args=_rt(cores, [[r[2] - r[1]] for r in rt]), config=cfg)
        prog = ttnn.ProgramDescriptor(kernels=[rd, wr, cp], semaphores=[], cbs=cbs)
        key = ("i2m_gmm", G, gs, K, N, E, nb, S, str(w.dtype), self.fid,
               tuple(repr(t.memory_config()) for t in (a, w, ids, out)))
        prog.custom_program_hash = int.from_bytes(hashlib.sha1(repr(key).encode()).digest()[:8], "little") & ((1 << 63) - 1)
        ttnn.generic_op([a, w, ids, out], prog)
        return out
