# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Grouped expert matmul for the sorted prefill MoE (tt/kernels/gmm_*.cpp): A [1, G, gs, K] bf16 (the dispatched
rows, groups sorted by expert), W [1, E, K, N] (this die's expert weights) and the groups' expert ids [1, G] uint16
-> C [1, G, gs, N] bf16. Core (b, s): output columns [b nb, (b + 1) nb) of the groups of a contiguous range s of the
experts (each expert's weight block read once per column block). Same math as ttnn.sparse_matmul with the whole
K in one in0 block (the program's fidelity, FP32 DST, K accumulated in order); ttnn's 1D config puts the N / 32 output
tiles of gate|up (24 at TP4) on that many cores, this one spreads (column block, expert range) over the grid."""

from __future__ import annotations

import hashlib
import struct

import ttnn

from .bconv import KDIR, _cb, _range_set, _rt

# I2M_GMM_DB (act mode; auto: when the experts average >= 3 groups, i.e. the compute per weight block is long): the
# weight block double-buffered (h = 2 columns: 922 KB of CBs at K 2304) so the next expert's weights load during the
# compute (bitwise; beside the packed prefill's L1 buffers: tests/test_packed_device.py output identical); 0 / 1:
# never / always
ACT_DB = __import__("os").environ.get("I2M_GMM_DB", "auto")
ACT_KB = 960
# I2M_GMM_WW: the writer loads the weight blocks on NOC 0 and the reader feeds A on NOC 1 (tt/kernels/gmm_writer.cpp),
# so the next expert's weights stream while A is multicast (the reader did both, one after the other); bitwise. auto:
# the gate|up (act) calls (the downs measured slower); 0 / 1: never / always
WW = __import__("os").environ.get("I2M_GMM_WW", "auto")
# I2M_GMM_D3: the down projections of all sub-shards in one program (gmm(..., out=[one per sub-shard])): each A row
# read and multicast once; bitwise
D3 = __import__("os").environ.get("I2M_GMM_D3", "1") == "1"
# I2M_GMM_BAL = R (0: off): the group slices' expert ranges split on device by cost (R per expert with groups + 1 per
# group, tt/kernels/gmm_bal.hpp) instead of equal expert counts, when the experts average >= 3 groups (one die: 2048-row
# chunks gmm 264 -> 241 ms; 1024: 180 -> 201, so off there); one die only (TP4 2048-row chunk 640 -> 880 ms); bitwise
BAL = int(__import__("os").environ.get("I2M_GMM_BAL", "5"))
TILE_BYTES = {ttnn.bfloat4_b: 576, ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}


def mc(lo, hi, ww):
    """the A multicast rectangle (on NOC 1 with ww: from the far corner)"""
    return [hi.x, hi.y, lo.x, lo.y] if ww else [lo.x, lo.y, hi.x, hi.y]


def _ww(act):
    return WW == "1" or (WW == "auto" and act)


def _nocs(ww):
    """the reader's and the writer's NOC"""
    return (ttnn.NOC.NOC_1, ttnn.NOC.NOC_0) if ww else (ttnn.NOC.NOC_0, ttnn.NOC.NOC_1)


class GroupedMM:
    def __init__(self, mesh, ckc_fidelity=ttnn.MathFidelity.HiFi2, max_cores=None):
        self.mesh, self.fid = mesh, ckc_fidelity
        g = mesh.compute_with_storage_grid_size()
        self.gx, self.ncores = g.x, max_cores or g.x * g.y  # (whole rows of the grid)

    def __call__(self, a, w, ids, out, E, nb=None, wsub=(1, 0), k0=0, act=None, nsub=1, koff=0, K=None, ocol=0):
        """a [1, G, gs, K] bf16, w [1, E, K, N] (bfp8 / bf16 tiles), ids [1, G] uint16 ROW_MAJOR, out [1, G, gs, N].
        wsub (S, s): w holds S sub-shards per expert as virtual experts e * S + s ([1, E * S, K, N]); this call's s.
        k0: w's experts have more K rows than a ([1, E, Kw, N]): rows [k0, k0 + K) of each.
        K, koff: the product reads a's columns [koff, koff + K) (default: all of them).
        act (the dispatched rows' routing weights [1, 1, Rd, 1] bf16): w is a gate|up weight ([g | u], W columns
        each) and C = bf16(bf16(tanh(g / 4) sigmoid(g) tanh(u)) w) [1, G, gs, nsub * W] (tt/moeact.py's math, bitwise),
        sub-shards s .. s + nsub - 1 of wsub side by side, from out's column tile ocol.
        out a list (no act): the products of sub-shards wsub[1] + i (out[i]) of a's columns koff + i K, one program."""
        G, gs, KA = int(a.shape[1]), int(a.shape[2]), int(a.shape[3])
        K = KA if K is None else K
        N = int(w.shape[-1])
        Kt, Nt, mt = K // 32, N // 32, gs // 32
        w_page = TILE_BYTES[w.dtype]
        outs = out if isinstance(out, list) else [out]
        NS = len(outs)
        rm = outs[0].layout == ttnn.ROW_MAJOR_LAYOUT  # (row-major C: untilized blocks of <= 4 tiles with FP32 DST)
        if act is not None:
            return self._act(a, w, ids, out, E, act, wsub, nsub, Kt, Nt, mt, w_page, G, gs, ocol)
        # the column block (DST: 8 FP32 tiles, Nt % nb == 0) and the weight blocks in the CB (2: the next expert's loads
        # during the compute): the weight and A CBs within ~800 KB of L1 (the packed prefill keeps L1 buffers above
        # ~1.3 MB); the widest block first (fewer A multicast receivers)
        fits = lambda d, k: NS * Kt * (k * d * w_page + 2 * 2048) <= 800 * 1024  # noqa: E731
        if nb is None:
            nb = max(d for d in range(1, 5 if rm else 9) if Nt % d == 0 and fits(d, 1))
        nbuf = 2 if fits(nb, 2) else 1
        NB = Nt // nb
        ww = _ww(False)
        # a slice's NB cores: a w x h rectangle (b = 0 at its corner multicasts the A tiles to the others)
        gy = self.ncores // self.gx
        # (the rectangle that fits the most slices)
        w_ = max((d for d in range(1, self.gx + 1) if NB % d == 0 and NB // d <= gy),
                 key=lambda d: (self.gx // d) * (gy // (NB // d)))
        h_ = NB // w_
        per_row = self.gx // w_
        S = max(1, min(E, per_row * (gy // h_)))  # slices: contiguous expert ranges (each expert's weights read once)
        bal = BAL if G >= 3 * E and self.mesh.get_num_devices() == 1 else 0
        bounds = [(s, S) if bal else (E * s // S, E * (s + 1) // S) for s in range(S)]
        cores, rt = [], []
        for s in range(S):
            x0, y0 = (s % per_row) * w_, (s // per_row) * h_
            snoc = self.mesh.worker_core_from_logical_core(ttnn.CoreCoord(x0, y0))
            hi = self.mesh.worker_core_from_logical_core(ttnn.CoreCoord(x0 + w_ - 1, y0 + h_ - 1))
            for b in range(NB):
                cores.append(ttnn.CoreCoord(x0 + b % w_, y0 + b // w_))
                rt.append([b, *bounds[s], snoc.x, snoc.y] + mc(snoc, hi, ww) + [G])
        crs = _range_set(cores)
        cb_a, cb_w, cb_ids, cb_out = 0, 1, 2, 3
        ids_bytes = (G * 2 + 63) // 64 * 64
        cbs = [_cb(crs, cb_a, ttnn.bfloat16, 2048, 2 * NS * Kt), _cb(crs, cb_w, w.dtype, w_page, nbuf * NS * Kt * nb),
               _cb(crs, cb_ids, ttnn.uint16, ids_bytes, 1), _cb(crs, cb_out, ttnn.bfloat16, 2048, 2 * nb)]
        dm = ttnn.DataMovementConfigDescriptor
        defs = [("GMM_RM", str(int(rm))), ("GMM_WW", str(int(ww))), ("GMM_NS", str(NS)), ("GMM_BAL", str(bal))]
        rd = ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "gmm_reader.cpp"), core_ranges=crs, defines=defs,
            compile_time_args=[cb_a, cb_w, cb_ids, Kt, Nt, nb, mt, E, ids_bytes, w_page, KA // 32, 0, 0, NB]
            + ttnn.TensorAccessorArgs(a).get_compile_time_args() + ttnn.TensorAccessorArgs(w).get_compile_time_args()
            + ttnn.TensorAccessorArgs(ids).get_compile_time_args() + ttnn.TensorAccessorArgs(ids).get_compile_time_args(),
            runtime_args=_rt(cores, [r + [0, r[0] * nb] for r in rt]),
            config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=_nocs(ww)[0]))
        rd.common_runtime_args = [a.buffer_address(), w.buffer_address(), ids.buffer_address(), wsub[0], wsub[1],
                                  int(w.shape[-2]) // 32, k0, koff // 32, 0]
        wr = ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "gmm_writer.cpp"), core_ranges=crs, defines=defs,
            compile_time_args=[cb_out, Nt, nb, mt, cb_ids, E] + ttnn.TensorAccessorArgs(outs[0]).get_compile_time_args()
            + [cb_w, Kt, Nt, w_page, 0] + ttnn.TensorAccessorArgs(w).get_compile_time_args(),
            runtime_args=_rt(cores, [r + [r[0] * nb, 0, r[0] * nb] for r in rt]),
            config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=_nocs(ww)[1]))
        wr.common_runtime_args = [w.buffer_address(), wsub[0], wsub[1], int(w.shape[-2]) // 32, k0] + [
            o.buffer_address() for o in outs]
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=self.fid, fp32_dest_acc_en=True, dst_full_sync_en=True,
                                           math_approx_mode=False)
        cp = ttnn.KernelDescriptor(kernel_source=str(KDIR / "gmm_compute.cpp"), core_ranges=crs, defines=defs,
                                   compile_time_args=[cb_a, cb_w, cb_out, Kt, nb, mt, cb_ids, E],
                                   runtime_args=_rt(cores, [[r[1], r[2], G] for r in rt]), config=cfg)
        sems = [ttnn.SemaphoreDescriptor(i, ttnn.CoreType.WORKER, crs, 0) for i in range(2)]
        prog = ttnn.ProgramDescriptor(kernels=[rd, wr, cp], semaphores=sems, cbs=cbs)
        key = ("i2m_gmm6", tuple(defs), G, gs, K, KA, N, E, nb, nbuf, S, rm, str(w.dtype), self.fid,
               tuple(repr(t.memory_config()) for t in [a, w, ids] + outs))
        prog.custom_program_hash = int.from_bytes(hashlib.sha1(repr(key).encode()).digest()[:8], "little") & ((1 << 63) - 1)
        ttnn.generic_op([a, w, ids] + outs, prog)
        return out

    def _slices(self, NB):
        """The group slices for NB column blocks: (S, cores' (x, y) per (slice, b), the slice corners)."""
        gy = self.ncores // self.gx
        w_ = max((d for d in range(1, self.gx + 1) if NB % d == 0 and NB // d <= gy),
                 key=lambda d: (self.gx // d) * (gy // (NB // d)))
        h_ = NB // w_
        per_row = self.gx // w_
        return w_, h_, per_row, gy

    def _act(self, a, w, ids, out, E, wts, wsub, nsub, Kt, Nt, mt, w_page, G, gs, ocol=0):
        """gate|up + the activation and routing weights (see __call__): column block b of the output = sub-shard
        s = b // (Wt / h), its columns [c0, c0 + h); the weight block holds the h gate and the h up columns."""
        Wt = Nt // 2
        # (DST: 2h products + the weights + a scratch tile <= 8). I2M_GMM_DB: the widest h whose weight block fits
        # twice in ACT_KB of CBs (the next expert's weights load during the compute)
        hs = [d for d in (3, 2, 1) if Wt % d == 0]
        fits = lambda h_, k, kb: Kt * (k * 2 * h_ * w_page + 2 * 2048) <= kb * 1024  # noqa: E731
        db = ACT_DB == "1" or (ACT_DB == "auto" and G >= 3 * E)
        h = next((d for d in hs if fits(d, 2, ACT_KB)), hs[0]) if db else hs[0]
        nb = 2 * h
        nbuf = 2 if fits(h, 2, ACT_KB if db else 800) else 1
        NB = nsub * Wt // h
        ww = _ww(True)
        w_, h_, per_row, gy = self._slices(NB)
        S = max(1, min(E, per_row * (gy // h_)))
        bal = BAL if G >= 3 * E and self.mesh.get_num_devices() == 1 else 0
        bounds = [(s, S) if bal else (E * s // S, E * (s + 1) // S) for s in range(S)]
        cores, rt, wrt = [], [], []
        for s in range(S):
            x0, y0 = (s % per_row) * w_, (s // per_row) * h_
            snoc = self.mesh.worker_core_from_logical_core(ttnn.CoreCoord(x0, y0))
            hi = self.mesh.worker_core_from_logical_core(ttnn.CoreCoord(x0 + w_ - 1, y0 + h_ - 1))
            for b in range(NB):
                cores.append(ttnn.CoreCoord(x0 + b % w_, y0 + b // w_))
                sub, c0 = b // (Wt // h), (b % (Wt // h)) * h
                rt.append([b, *bounds[s], snoc.x, snoc.y] + mc(snoc, hi, ww) + [G, sub, c0])
                wrt.append([b, *bounds[s], 0, 0, 0, 0, 0, 0, G, ocol + sub * Wt + c0, sub, c0])
        crs = _range_set(cores)
        cb_a, cb_w, cb_ids, cb_out, cb_rw = 0, 1, 2, 3, 4
        ids_bytes = (G * 2 + 63) // 64 * 64
        bf = ttnn.bfloat16
        cbs = [_cb(crs, cb_a, bf, 2048, 2 * Kt), _cb(crs, cb_w, w.dtype, w_page, nbuf * Kt * nb),
               _cb(crs, cb_ids, ttnn.uint16, ids_bytes, 1), _cb(crs, cb_out, bf, 2048, 2 * h),
               _cb(crs, cb_rw, bf, 2048, 2)]
        dm = ttnn.DataMovementConfigDescriptor
        defs = [("GMM_RM", "0"), ("GMM_ACT", "1"), ("GMM_WW", str(int(ww))), ("GMM_BAL", str(bal))]
        KA = int(a.shape[3]) // 32
        rd = ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "gmm_reader.cpp"), core_ranges=crs, defines=defs,
            compile_time_args=[cb_a, cb_w, cb_ids, Kt, Nt, nb, mt, E, ids_bytes, w_page, KA, cb_rw, Wt, NB]
            + ttnn.TensorAccessorArgs(a).get_compile_time_args() + ttnn.TensorAccessorArgs(w).get_compile_time_args()
            + ttnn.TensorAccessorArgs(ids).get_compile_time_args() + ttnn.TensorAccessorArgs(wts).get_compile_time_args(),
            runtime_args=_rt(cores, rt), config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=_nocs(ww)[0]))
        rd.common_runtime_args = [a.buffer_address(), w.buffer_address(), ids.buffer_address(), wsub[0], wsub[1],
                                  int(w.shape[-2]) // 32, 0, 0, wts.buffer_address()]
        No = int(out.shape[-1]) // 32
        wr = ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "gmm_writer.cpp"), core_ranges=crs, defines=defs,
            compile_time_args=[cb_out, No, h, mt, cb_ids, E] + ttnn.TensorAccessorArgs(out).get_compile_time_args()
            + [cb_w, Kt, Nt, w_page, Wt] + ttnn.TensorAccessorArgs(w).get_compile_time_args(),
            runtime_args=_rt(cores, wrt), config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=_nocs(ww)[1]))
        wr.common_runtime_args = [w.buffer_address(), wsub[0], wsub[1], int(w.shape[-2]) // 32, 0, out.buffer_address()]
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=self.fid, fp32_dest_acc_en=True, dst_full_sync_en=True,
                                           math_approx_mode=False)
        quarter = struct.unpack("<I", struct.pack("<f", 0.25))[0]
        cp = ttnn.KernelDescriptor(kernel_source=str(KDIR / "gmm_compute.cpp"), core_ranges=crs, defines=defs,
                                   compile_time_args=[cb_a, cb_w, cb_out, Kt, nb, mt, cb_ids, E, cb_rw, quarter],
                                   runtime_args=_rt(cores, [[r[1], r[2], G] for r in rt]), config=cfg)
        sems = [ttnn.SemaphoreDescriptor(i, ttnn.CoreType.WORKER, crs, 0) for i in range(2)]
        prog = ttnn.ProgramDescriptor(kernels=[rd, wr, cp], semaphores=sems, cbs=cbs)
        key = ("i2m_gmm_act1", tuple(defs), G, gs, Kt, KA, Nt, No, E, h, nbuf, S, nsub, str(w.dtype), self.fid,
               tuple(repr(t.memory_config()) for t in (a, w, ids, out, wts)))
        prog.custom_program_hash = int.from_bytes(hashlib.sha1(repr(key).encode()).digest()[:8], "little") & ((1 << 63) - 1)
        ttnn.generic_op([a, w, ids, out, wts], prog)
        return out
