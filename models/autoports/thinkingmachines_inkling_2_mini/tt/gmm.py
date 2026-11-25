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
from .bconv import _ks

# I2M_GMM_DB (act mode; auto: when the experts average >= 3 groups, i.e. the compute per weight block is long, or with
# I2M_GMM_BANDS: one die 512-row chunks gate|up 1.81 -> 1.68 ms per application, 6 slices of 18 cores, not 8 of 12): the
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
# I2M_GMM_GS (act: default): group slices take equal counts of the real groups (tt/kernels/gmm_bal.hpp gmm_gs; an
# expert's groups may span slices, each loads its weight block) instead of contiguous expert ranges: a hot expert (a
# random-token 2048-row chunk routes up to 64 groups to one expert) no longer stalls its slice. act: the gate|up calls
# (bench, gate|up per application: TP4 2048 rows 1.27 -> 0.96 + rem 0.52 -> 0.41 ms, one die 3.78 -> 3.51 ms vs BAL);
# 1: the downs too (measured 3-7 % slower); 0: off. Replaces BAL; bitwise (the same per-group products)
GS = __import__("os").environ.get("I2M_GMM_GS", "act")
# I2M_GMM_BANDS (auto: one die): the group slices as wide horizontal bands down the grid's left columns + tall slices in
# the columns left (one die, 18-core slices: 6 slices on 108 of the 110 cores, not 5 2 x 9 slices on 90), the act
# calls' GS group counts weighted per slice (I2M_GMM_SHARES: integers in slice order; auto: by the slice's top grid
# row, the weight streams on NOC 0 run slower toward the top), and the A multicast's sender at the rectangle corner
# its NOC starts from (NOC 1: the bottom right; from the top left its multicasts wrapped around the grid). Bitwise
# (the same per-group products); one die 2048-row chunks: gate|up 3.45 -> 2.35, downs 1.66 -> 1.53 ms per application
_ENV = __import__("os").environ
BANDS = _ENV.get("I2M_GMM_BANDS", "auto")
SHARES = _ENV.get("I2M_GMM_SHARES", "auto")
# I2M_PCACHE (default 1): custom programs keep their kernel / CB descriptors per shape key and only set the per-call
# common runtime args (buffer addresses) on a hit: the same programs, less host time per call
PCACHE = _ENV.get("I2M_PCACHE", "1") == "1"
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
        self._pc = {}  # (PCACHE) structural key -> the call's descriptors but its common runtime args

    def __call__(self, a, w, ids, out, E, nb=None, wsub=(1, 0), k0=0, act=None, nsub=1, koff=0, K=None, ocol=0,
                 w2=None):
        """a [1, G, gs, K] bf16, w [1, E, K, N] (bfp8 / bf16 tiles), ids [1, G] uint16 ROW_MAJOR, out [1, G, gs, N].
        wsub (S, s): w holds S sub-shards per expert as virtual experts e * S + s ([1, E * S, K, N]); this call's s.
        k0: w's experts have more K rows than a ([1, E, Kw, N]): rows [k0, k0 + K) of each.
        K, koff: the product reads a's columns [koff, koff + K) (default: all of them).
        act (the dispatched rows' routing weights [1, 1, Rd, 1] bf16): w is a gate|up weight ([g | u], W columns
        each) and C = bf16(bf16(tanh(g / 4) sigmoid(g) tanh(u)) w) [1, G, gs, nsub * W] (tt/moeact.py's math, bitwise),
        sub-shards s .. s + nsub - 1 of wsub side by side, from out's column tile ocol.
        out a list (no act): the products of sub-shards wsub[1] + i (out[i]) of a's columns koff + i K, one program.
        w2 (act, one sub-shard): a second gate|up weight [1, E, K, 2 W2] whose W2 column pairs follow w's in the output
        (the REM layout's remainder pairs: one program for the die's whole slice, no padding)."""
        G, gs, KA = int(a.shape[1]), int(a.shape[2]), int(a.shape[3])
        K = KA if K is None else K
        N = int(w.shape[-1])
        Kt, Nt, mt = K // 32, N // 32, gs // 32
        w_page = TILE_BYTES[w.dtype]
        outs = out if isinstance(out, list) else [out]
        NS = len(outs)
        rm = outs[0].layout == ttnn.ROW_MAJOR_LAYOUT  # (row-major C: untilized blocks of <= 4 tiles with FP32 DST)
        if act is not None:
            return self._act(a, w, ids, out, E, act, wsub, nsub, Kt, Nt, mt, w_page, G, gs, ocol, w2)
        ck = ("mm", G, gs, K, KA, N, E, nb, NS, rm, w.dtype, tuple(repr(t.memory_config()) for t in [a, w, ids] + outs))
        hit = self._pc.get(ck) if PCACHE else None
        if hit is None:
            hit = self._pc[ck] = self._mm_desc(a, w, ids, outs, E, nb, Kt, Nt, mt, w_page, G, gs, KA, K, N, NS, rm)
        rd, wr, rest = hit
        rd.common_runtime_args = [a.buffer_address(), w.buffer_address(), ids.buffer_address(), wsub[0], wsub[1],
                                  int(w.shape[-2]) // 32, k0, koff // 32, 0]
        wr.common_runtime_args = [w.buffer_address(), wsub[0], wsub[1], int(w.shape[-2]) // 32, k0] + [
            o.buffer_address() for o in outs]
        prog = ttnn.ProgramDescriptor(kernels=[rd, wr, rest[0]], semaphores=rest[1], cbs=rest[2])
        prog.custom_program_hash = rest[3]
        ttnn.generic_op([a, w, ids] + outs, prog)
        return out

    def _mm_desc(self, a, w, ids, outs, E, nb, Kt, Nt, mt, w_page, G, gs, KA, K, N, NS, rm):
        """__call__'s (no act) kernels, semaphores, CBs and hash: everything but the common runtime args (addresses,
        wsub, k0, koff), which the call sets"""
        # the column block (DST: 8 FP32 tiles, Nt % nb == 0) and the weight blocks in the CB (2: the next expert's loads
        # during the compute): the weight and A CBs within ~800 KB of L1 (the packed prefill keeps L1 buffers above
        # ~1.3 MB); the widest block first (fewer A multicast receivers)
        fits = lambda d, k: NS * Kt * (k * d * w_page + 2 * 2048) <= 800 * 1024  # noqa: E731
        if nb is None:
            nb = max(d for d in range(1, 5 if rm else 9) if Nt % d == 0 and fits(d, 1))
        nbuf = 2 if fits(nb, 2) else 1
        NB = Nt // nb
        ww = _ww(False)
        gs_ = GS == "1"
        bal = BAL if G >= 3 * E and self.mesh.get_num_devices() == 1 and not gs_ else 0
        S, cores, rt = self._place(NB, E, ww, bal, gs_)
        rt = [r + [G] for r in rt]
        crs = _range_set(cores)
        cb_a, cb_w, cb_ids, cb_out = 0, 1, 2, 3
        ids_bytes = (G * 2 + 63) // 64 * 64
        cbs = [_cb(crs, cb_a, ttnn.bfloat16, 2048, 2 * NS * Kt), _cb(crs, cb_w, w.dtype, w_page, nbuf * NS * Kt * nb),
               _cb(crs, cb_ids, ttnn.uint16, ids_bytes, 1), _cb(crs, cb_out, ttnn.bfloat16, 2048, 2 * nb)]
        dm = ttnn.DataMovementConfigDescriptor
        defs = [("GMM_RM", str(int(rm))), ("GMM_WW", str(int(ww))), ("GMM_NS", str(NS)), ("GMM_BAL", str(bal)),
                ("GMM_GS", str(int(gs_)))]
        rd = ttnn.KernelDescriptor(
            kernel_source=_ks(KDIR, "gmm_reader.cpp"), core_ranges=crs, defines=defs,
            compile_time_args=[cb_a, cb_w, cb_ids, Kt, Nt, nb, mt, E, ids_bytes, w_page, KA // 32, 0, 0, NB]
            + ttnn.TensorAccessorArgs(a).get_compile_time_args() + ttnn.TensorAccessorArgs(w).get_compile_time_args()
            + ttnn.TensorAccessorArgs(ids).get_compile_time_args() + ttnn.TensorAccessorArgs(ids).get_compile_time_args(),
            runtime_args=_rt(cores, [r + [0, r[0] * nb] for r in rt]),
            config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=_nocs(ww)[0]))
        wr = ttnn.KernelDescriptor(
            kernel_source=_ks(KDIR, "gmm_writer.cpp"), core_ranges=crs, defines=defs,
            compile_time_args=[cb_out, Nt, nb, mt, cb_ids, E] + ttnn.TensorAccessorArgs(outs[0]).get_compile_time_args()
            + [cb_w, Kt, Nt, w_page, 0] + ttnn.TensorAccessorArgs(w).get_compile_time_args(),
            runtime_args=_rt(cores, [r + [r[0] * nb, 0, r[0] * nb] for r in rt]),
            config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=_nocs(ww)[1]))
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=self.fid, fp32_dest_acc_en=True, dst_full_sync_en=True,
                                           math_approx_mode=False)
        cp = ttnn.KernelDescriptor(kernel_source=_ks(KDIR, "gmm_compute.cpp"), core_ranges=crs, defines=defs,
                                   compile_time_args=[cb_a, cb_w, cb_out, Kt, nb, mt, cb_ids, E],
                                   runtime_args=_rt(cores, [[r[1], r[2], G] for r in rt]), config=cfg)
        sems = [ttnn.SemaphoreDescriptor(i, ttnn.CoreType.WORKER, crs, 0) for i in range(2)]
        key = ("i2m_gmm6", tuple(defs), G, gs, K, KA, N, E, nb, nbuf, S, rm, str(w.dtype), self.fid, self._lkey(NB),
               tuple(repr(t.memory_config()) for t in [a, w, ids] + outs))
        h = int.from_bytes(hashlib.sha1(repr(key).encode()).digest()[:8], "little") & ((1 << 63) - 1)
        return rd, wr, (cp, sems, cbs, h)

    def _bands(self):
        return BANDS == "1" or (BANDS == "auto" and self.mesh.get_num_devices() == 1)

    def _rects(self, NB):
        """The group slices' rectangles (x0, y0, w, h) of NB cores: one shape tiled over the grid (the shape that fits
        the most), or (bands) wide horizontal bands down the grid's left columns + the shape that fits the most in the
        columns left."""
        gx, gy = self.gx, self.ncores // self.gx
        shapes = [(d, NB // d) for d in range(1, gx + 1) if NB % d == 0 and NB // d <= gy]
        fit = lambda w, h, sh: max(sh, key=lambda s: ((w // s[0]) * (h // s[1])))  # noqa: E731
        if not self._bands():
            w_, h_ = fit(gx, gy, shapes)
            per_row = gx // w_
            return [((i % per_row) * w_, (i // per_row) * h_, w_, h_) for i in range(per_row * (gy // h_))]
        best = []
        for w1, h1 in sorted(shapes, reverse=True):  # (the widest bands first)
            r = [(0, y, w1, h1) for y in range(0, gy - h1 + 1, h1)]
            left, sub = gx - w1, [s for s in shapes if s[0] <= gx - w1]
            if sub:
                w2, h2 = fit(left, gy, sub)
                r += [(w1 + i * w2, j * h2, w2, h2) for j in range(gy // h2) for i in range(left // w2)]
            if len(r) > len(best):
                best = r
        return best

    def _shares(self, rects):
        """The slices' GS weights (bands: the measured one die rates of the 18-core layout, else 41 + 2 y0)."""
        if not self._bands() or SHARES in ("0", ""):
            return [1] * len(rects)
        if SHARES != "auto":
            return ([int(v) for v in SHARES.split(",")] + [1] * len(rects))[:len(rects)]
        if rects == [(0, 0, 9, 2), (0, 2, 9, 2), (0, 4, 9, 2), (0, 6, 9, 2), (0, 8, 9, 2), (9, 0, 2, 9)]:
            return [41, 45, 48, 54, 57, 46]
        return [41 + 2 * y0 for _, y0, _, _ in rects]

    def _lkey(self, NB):
        return tuple(self._rects(NB)), SHARES

    def _place(self, NB, E, ww, bal, gs_):
        """The group slices' cores: (S, cores, per core [b, e0, e1, sender x, y, multicast rectangle (4)]). GS: e0, e1
        = the slice's cumulative share q0, Q << 16 | q1 (tt/kernels/gmm_bal.hpp gmm_gs)."""
        rects = self._rects(NB)
        S = max(1, min(E, len(rects)))
        rects = rects[:S]
        sh = self._shares(rects)
        cum = [sum(sh[:i]) for i in range(S + 1)]
        cores, rt = [], []
        for s, (x0, y0, w_, h_) in enumerate(rects):
            lo = self.mesh.worker_core_from_logical_core(ttnn.CoreCoord(x0, y0))
            hi = self.mesh.worker_core_from_logical_core(ttnn.CoreCoord(x0 + w_ - 1, y0 + h_ - 1))
            rev = ww and self._bands()  # (the NOC 1 multicast's sender at the bottom right: b counts from there)
            if gs_:
                bounds = (cum[s], cum[S] << 16 | cum[s + 1])
            elif bal:
                bounds = (s, S)
            else:
                bounds = (E * s // S, E * (s + 1) // S)
            snoc = hi if rev else lo
            for b in range(NB):
                i = NB - 1 - b if rev else b
                cores.append(ttnn.CoreCoord(x0 + i % w_, y0 + i // w_))
                rt.append([b, *bounds, snoc.x, snoc.y] + mc(lo, hi, ww))
        return S, cores, rt

    def _act(self, a, w, ids, out, E, wts, wsub, nsub, Kt, Nt, mt, w_page, G, gs, ocol=0, w2=None):
        """gate|up + the activation and routing weights (see __call__): column block b of the output = sub-shard
        s = b // (Wt / h), its columns [c0, c0 + h); the weight block holds the h gate and the h up columns.
        w2: pairs Wt .. Wt + W2 - 1 come from w2 (nsub 1)."""
        ck = ("act", G, gs, Kt, int(a.shape[3]), Nt, E, nsub, ocol, w.dtype, int(out.shape[-1]),
              None if w2 is None else (tuple(w2.shape), w2.dtype),
              tuple(repr(t.memory_config()) for t in (a, w, ids, out, wts)))
        hit = self._pc.get(ck) if PCACHE else None
        if hit is None:
            hit = self._pc[ck] = self._act_desc(a, w, ids, out, E, wts, nsub, Kt, Nt, mt, w_page, G, gs, ocol, w2)
        rd, wr, rest = hit
        assert w2 is None or wsub == (1, 0)
        w2a = [w2.buffer_address()] if w2 is not None else []
        rd.common_runtime_args = [a.buffer_address(), w.buffer_address(), ids.buffer_address(), wsub[0], wsub[1],
                                  int(w.shape[-2]) // 32, 0, 0, wts.buffer_address()] + w2a
        wr.common_runtime_args = [w.buffer_address(), wsub[0], wsub[1], int(w.shape[-2]) // 32, 0,
                                  out.buffer_address()] + w2a
        prog = ttnn.ProgramDescriptor(kernels=[rd, wr, rest[0]], semaphores=rest[1], cbs=rest[2])
        prog.custom_program_hash = rest[3]
        ttnn.generic_op([a, w, ids, out, wts] + ([w2] if w2 is not None else []), prog)
        return out

    def _act_desc(self, a, w, ids, out, E, wts, nsub, Kt, Nt, mt, w_page, G, gs, ocol, w2):
        """_act's kernels, semaphores, CBs and hash (the common runtime args set per call)"""
        Wt = Nt // 2
        W2 = 0
        if w2 is not None:
            assert nsub == 1 and w2.dtype == w.dtype and int(w2.shape[-2]) == int(w.shape[-2])
            assert repr(w2.memory_config()) == repr(w.memory_config())
            assert w.memory_config().buffer_type == ttnn.BufferType.DRAM  # (interleaved: one accessor's args for both)
            W2 = int(w2.shape[-1]) // 64
        P = Wt + W2  # (the output's column pairs per sub-shard)
        # (DST: 2h products + the weights + a scratch tile <= 8). I2M_GMM_DB: the widest h whose weight block fits
        # twice in ACT_KB of CBs (the next expert's weights load during the compute)
        # (auto: not narrower than half the widest h: TP4's 9 pairs ran faster at h = 3 single-buffered than at h = 1
        # double-buffered)
        hs = [d for d in (3, 2, 1) if P % d == 0]
        fits = lambda h_, k, kb: Kt * (k * 2 * h_ * w_page + 2 * 2048) <= kb * 1024  # noqa: E731
        db = ACT_DB == "1" or (ACT_DB == "auto" and (G >= 3 * E or self._bands()))
        h = next((d for d in hs if fits(d, 2, ACT_KB) and (ACT_DB == "1" or 2 * d >= hs[0])), hs[0]) if db else hs[0]
        nb = 2 * h
        nbuf = 2 if fits(h, 2, ACT_KB if db else 800) else 1
        NB = nsub * P // h
        ww = _ww(True)
        gs_ = GS in ("1", "act")
        bal = BAL if G >= 3 * E and self.mesh.get_num_devices() == 1 and not gs_ else 0
        S, cores, prt = self._place(NB, E, ww, bal, gs_)
        rt, wrt = [], []
        for r in prt:
            b = r[0]
            sub, c0 = b // (P // h), (b % (P // h)) * h
            rt.append(r + [G, sub, c0])
            wrt.append([b, r[1], r[2], 0, 0, 0, 0, 0, 0, G, ocol + sub * P + c0, sub, c0])
        crs = _range_set(cores)
        cb_a, cb_w, cb_ids, cb_out, cb_rw = 0, 1, 2, 3, 4
        ids_bytes = (G * 2 + 63) // 64 * 64
        bf = ttnn.bfloat16
        cbs = [_cb(crs, cb_a, bf, 2048, 2 * Kt), _cb(crs, cb_w, w.dtype, w_page, nbuf * Kt * nb),
               _cb(crs, cb_ids, ttnn.uint16, ids_bytes, 1), _cb(crs, cb_out, bf, 2048, 2 * h),
               _cb(crs, cb_rw, bf, 2048, 2)]
        dm = ttnn.DataMovementConfigDescriptor
        defs = [("GMM_RM", "0"), ("GMM_ACT", "1"), ("GMM_WW", str(int(ww))), ("GMM_BAL", str(bal)),
                ("GMM_GS", str(int(gs_))), ("GMM_W2", str(W2))]
        KA = int(a.shape[3]) // 32
        rd = ttnn.KernelDescriptor(
            kernel_source=_ks(KDIR, "gmm_reader.cpp"), core_ranges=crs, defines=defs,
            compile_time_args=[cb_a, cb_w, cb_ids, Kt, Nt, nb, mt, E, ids_bytes, w_page, KA, cb_rw, Wt, NB]
            + ttnn.TensorAccessorArgs(a).get_compile_time_args() + ttnn.TensorAccessorArgs(w).get_compile_time_args()
            + ttnn.TensorAccessorArgs(ids).get_compile_time_args() + ttnn.TensorAccessorArgs(wts).get_compile_time_args(),
            runtime_args=_rt(cores, rt), config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=_nocs(ww)[0]))
        No = int(out.shape[-1]) // 32
        wr = ttnn.KernelDescriptor(
            kernel_source=_ks(KDIR, "gmm_writer.cpp"), core_ranges=crs, defines=defs,
            compile_time_args=[cb_out, No, h, mt, cb_ids, E] + ttnn.TensorAccessorArgs(out).get_compile_time_args()
            + [cb_w, Kt, Nt, w_page, Wt] + ttnn.TensorAccessorArgs(w).get_compile_time_args(),
            runtime_args=_rt(cores, wrt), config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=_nocs(ww)[1]))
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=self.fid, fp32_dest_acc_en=True, dst_full_sync_en=True,
                                           math_approx_mode=False)
        quarter = struct.unpack("<I", struct.pack("<f", 0.25))[0]
        cp = ttnn.KernelDescriptor(kernel_source=_ks(KDIR, "gmm_compute.cpp"), core_ranges=crs, defines=defs,
                                   compile_time_args=[cb_a, cb_w, cb_out, Kt, nb, mt, cb_ids, E, cb_rw, quarter],
                                   runtime_args=_rt(cores, [[r[1], r[2], G] for r in rt]), config=cfg)
        sems = [ttnn.SemaphoreDescriptor(i, ttnn.CoreType.WORKER, crs, 0) for i in range(2)]
        key = ("i2m_gmm_act1", tuple(defs), G, gs, Kt, KA, Nt, No, E, h, nbuf, S, nsub, str(w.dtype), self.fid,
               self._lkey(NB),
               tuple(repr(t.memory_config()) for t in (a, w, ids, out, wts)), W2)
        return rd, wr, (cp, sems, cbs,
                        int.from_bytes(hashlib.sha1(repr(key).encode()).digest()[:8], "little") & ((1 << 63) - 1))
