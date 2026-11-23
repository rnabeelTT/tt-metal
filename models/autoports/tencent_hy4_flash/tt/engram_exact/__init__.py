# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Engram decode tail with the default chain's numerics (tt/engram.Hy4EngramDevice._forward_decode, HY4_ENG_FUSED):
the ~20 elementwise ttnn programs around the two reductions folded into 4 generic_ops, the reductions themselves
(ttnn.rms_norm over [v | k_0..3 | x_0..3], engram._row_sum of the gate dots) kept as ttnn calls:

    pre   allv = [fp32(bf16(y + b)) ; fp32(X)]                         (ttnn.add -> typecast, typecast, concat)
          n    = ttnn.rms_norm(allv)
    mid   nw = n * w; main = bf16(nw_0); kq_i = nw_{1+i} nw_{5+i}       (multiply, slice, typecast, multiply)
    fold  (> 8 rows) S_i = ((kq_i,0 + kq_i,1) + ...) + kq_i,95 tile by tile; s = engram._row_sum(S): the 8-row-block sums
          of _row_sum then re-sum one tile, not 96 (R = 32: 326 -> 102 us); <= 8 rows: s = engram._row_sum(kq)
    gate  g_i  = bf16(sigmoid(sign(s c) sqrt(max(|s c|, 1e-6))))       (multiply, abs, clamp, sqrt, sign, ...)
    tail  X_i' = X_i + (sc_i + g_i) main                                (binary_ng bf16: FPU add, SFPU mul, FPU add)

Every fused op is elementwise and uses the same LLK function, operand order and DST format as the ttnn op it
replaces (FP32 SFPU add / mul are IEEE RNE, the fp32 -> bf16 typecast RNE; binary_ng's bf16 add runs on the FPU into a BF16
DST (not RNE: ~10% of sums 1 ulp off RNE), its bf16 multiply on the SFPU; the tail does the same): the output
is bit-identical to the chain (tests/test_eng_exact.py; in-model: b8 / g8l / dec / dec12 dev outputs, engine A/B logits).
HY4_ENG_NC (default 96): cores of pre / mid / tail, one hidden tile each (traced, per Engram layer at 1 row: chain
285 us -> 242 us; 32 rows 622 -> 339 us).
Why the reductions stay in ttnn: the earlier one-program kernel (FP32 hi / lo matmul partials over 32 cores) summed
in another order / precision than rms_norm's TF32 tile reduction and ttnn.sum, moved the Engram gate by bf16 ulps
and the b8 gate's KL 0.2229 -> 0.2365 (journal:hy4-fuse:0016)."""

from __future__ import annotations

import os
import struct
from pathlib import Path

import ttnn

KDIR = Path(__file__).parent / "kernels"
TILE = 32


def _bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


class EngramExactOp:
    _OPS = {}

    @classmethod
    def get(cls, mesh, hidden):
        key = (id(mesh), hidden)
        if key not in cls._OPS:
            cls._OPS[key] = cls(mesh, hidden)
        return cls._OPS[key]

    def __init__(self, mesh, hidden, nc=None):
        self.mesh, self.H = mesh, hidden
        self.HT = hidden // TILE
        if nc is None:  # HY4_ENG_NC (default 96: one hidden tile per core; 32: three)
            nc = int(os.environ.get("HY4_ENG_NC", "96"))
        g = mesh.compute_with_storage_grid_size()
        assert self.HT % nc == 0 and nc <= g.x * g.y
        self.NC, self.J = nc, self.HT // nc
        w = 8 if nc % 8 == 0 and nc // 8 <= g.y else g.x  # row-major over the grid (a partial last row)
        self.cores = [ttnn.CoreCoord(i % w, i // w) for i in range(nc)]
        full, rem = divmod(nc, w)
        rs = [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(w - 1, full - 1))] if full else []
        if rem:
            rs.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(rem - 1, full)))
        self.crs = ttnn.CoreRangeSet(rs)
        self.cores4 = self.cores[:4]
        self.crs4 = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))])

    # ---- program assembly
    @staticmethod
    def _cb(crs, i, dt, n):
        page = {ttnn.bfloat16: 2048, ttnn.float32: 4096}[dt]
        fmt = ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)
        return ttnn.CBDescriptor(total_size=page * n, core_ranges=crs, format_descriptors=[fmt])

    @staticmethod
    def _rt(cores, args):
        r = ttnn.RuntimeArgs()
        for c, a in zip(cores, args):
            r[c.x][c.y] = a
        return r

    def _run(self, ins, outs, compute, cp_ct, cbs, fp32, cores, crs, tensors):
        """ins: up to 4 (tensor, nvec, mode, cb); outs: up to 2 (tensor, nvec, mode, cb)."""
        acc = lambda t: ttnn.TensorAccessorArgs(t).get_compile_time_args()  # noqa: E731
        page = lambda t: 4096 if t.dtype == ttnn.float32 else 2048  # noqa: E731
        ins = list(ins) + [None] * (4 - len(ins))
        outs = list(outs) + [None] * (2 - len(outs))
        any_t = tensors[0]
        rd_ct = [self.J, self.HT] + sum((acc(i[0] if i else any_t) for i in ins), [])
        wr_ct = [self.J, self.HT] + sum((acc(o[0] if o else any_t) for o in outs), [])

        def spec(e):
            if e is None:
                return [0, 0, 0, 3, 0]
            t, nvec, mode, cb = e
            return [t.buffer_address(), page(t), nvec, mode, cb]

        rd_rt = [[k] + sum((spec(e) for e in ins), []) for k in range(len(cores))]
        wr_rt = [[k] + sum((spec(e) for e in outs), []) for k in range(len(cores))]
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32,
                                           math_approx_mode=False)
        if fp32:
            modes = [ttnn.UnpackToDestMode.Default] * 64
            for i, dt, _ in cbs:
                if dt == ttnn.float32:
                    modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
            cfg.unpack_to_dest_mode = modes
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "ex_reader.cpp"), core_ranges=crs, compile_time_args=rd_ct,
                                  runtime_args=self._rt(cores, rd_rt),
                                  config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1)),
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "ex_writer.cpp"), core_ranges=crs, compile_time_args=wr_ct,
                                  runtime_args=self._rt(cores, wr_rt),
                                  config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0)),
            ttnn.KernelDescriptor(kernel_source=str(KDIR / compute), core_ranges=crs, compile_time_args=cp_ct,
                                  runtime_args=self._rt(cores, [[k] for k in range(len(cores))]), config=cfg),
        ]
        cbd = [self._cb(crs, i, dt, n) for i, dt, n in cbs]
        ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbd))

    def _alloc(self, shape, dt, mem):
        return ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, self.mesh, mem)

    # ---- the four programs
    def pre(self, y, yb, X, mem):
        """y [1, 1, 32, 5H] fp32, yb [1, 1, 1, 5H] fp32, X [1, 4, 32, H] bf16 -> allv [1, 9, 32, H] fp32."""
        J, F, B = self.J, ttnn.float32, ttnn.bfloat16
        out = self._alloc([1, 9, TILE, self.H], F, mem)
        cbs = [(0, F, 5 * J), (1, F, 5 * J), (2, B, 4 * J), (16, F, 9 * J)]
        self._run([(y, 5, 0, 0), (yb, 5, 0, 1), (X, 4, 0, 2)], [(out, 9, 0, 16)], "ex_pre_compute.cpp", [J], cbs,
                  True, self.cores, self.crs, [y, yb, X, out])
        return out

    def mid(self, n, w, mem):
        """n [1, 9, 32, H] fp32 (the rms_norm), w [1, 9, 1, H] fp32 -> main [1, 1, 32, H] bf16, kq [1, 4, 32, H] fp32."""
        J, F, B = self.J, ttnn.float32, ttnn.bfloat16
        main = self._alloc([1, 1, TILE, self.H], B, mem)
        kq = self._alloc([1, 4, TILE, self.H], F, mem)
        cbs = [(0, F, 9 * J), (1, F, 9 * J), (16, B, J), (17, F, 4 * J)]
        self._run([(n, 9, 0, 0), (w, 9, 0, 1)], [(main, 1, 0, 16), (kq, 4, 0, 17)], "ex_mid_compute.cpp", [J], cbs,
                  True, self.cores, self.crs, [n, w, main, kq])
        return main, kq

    def fold(self, kq, mem, blk=8):
        """kq [1, 4, 32, H] fp32 -> S [1, 4, 32, 32] fp32: the H/32 tiles of each stream summed elementwise in tile order
        (IEEE, the cross-tile fold of ttnn.sum's accurate fp32 reduce; one core per stream). engram._row_sum(S) then
        equals engram._row_sum(kq) bit for bit (same fold, the same in-tile row sum on the one remaining tile)."""
        F = ttnn.float32
        HT = self.HT
        blk = max(b for b in range(1, blk + 1) if HT % b == 0)
        out = self._alloc([1, 4, TILE, TILE], F, mem)
        crs, cores = self.crs4, self.cores4
        acc = lambda t: ttnn.TensorAccessorArgs(t).get_compile_time_args()  # noqa: E731
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           math_approx_mode=False)
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[0] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = modes
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "ex_fold_reader.cpp"), core_ranges=crs,
                                  compile_time_args=[HT, blk] + acc(kq),
                                  runtime_args=self._rt(cores, [[kq.buffer_address(), i] for i in range(4)]),
                                  config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1)),
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "ex_writer.cpp"), core_ranges=crs,
                                  compile_time_args=[1, HT] + acc(out) + acc(out),
                                  runtime_args=self._rt(cores, [[i, out.buffer_address(), 4096, 1, 2, 16, 0, 0, 0, 3, 0]
                                                                for i in range(4)]),
                                  config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0)),
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "ex_fold_compute.cpp"), core_ranges=crs,
                                  compile_time_args=[HT, blk], runtime_args=self._rt(cores, [[] for _ in range(4)]),
                                  config=cfg),
        ]
        cbs = [self._cb(crs, 0, F, 2 * blk), self._cb(crs, 16, F, 1)]
        ttnn.generic_op([kq, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return out

    def gate(self, s, inv_sqrt_h, mem):
        """s [1, 4, 32, 1] fp32 (the row sums) -> g [1, 4, 32, 1] bf16 (column 0 meaningful)."""
        F, B = ttnn.float32, ttnn.bfloat16
        g = self._alloc([1, 4, TILE, 1], B, mem)
        cbs = [(0, F, 1), (16, B, 1)]
        self._run([(s, 1, 2, 0)], [(g, 1, 2, 16)], "ex_gate_compute.cpp", [_bits(inv_sqrt_h), _bits(1e-6)], cbs,
                  True, self.cores4, self.crs4, [s, g])
        return g

    def tail(self, g, sc, main, X, mem, dbg=0):
        """g [1, 4, 32, 1] bf16, sc [1, 4, 1, H] bf16, main [1, 1, 32, H] bf16, X [1, 4, 32, H] bf16 -> X'."""
        J, B = self.J, ttnn.bfloat16
        out = self._alloc([1, 4, TILE, self.H], B, mem)
        cbs = [(0, B, 4), (1, B, 4 * J), (2, B, J), (3, B, 4 * J), (4, B, 4), (5, B, 1), (6, B, 1), (7, B, 1),
               (16, B, 4 * J)]
        self._run([(g, 4, 1, 0), (sc, 4, 0, 1), (main, 1, 0, 2), (X, 4, 0, 3)], [(out, 4, 0, 16)],
                  "ex_tail_compute.cpp", [J, dbg], cbs, False, self.cores, self.crs, [g, sc, main, X, out])
        return out
