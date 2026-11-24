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
from .post import RSPLIT, ZLAZY, red_k, red_sel

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


class _Rect:
    """A core rectangle for GateKernel2: origin (x0, y0), CX x CY cores, Ht / C hidden tiles each; its multicast range
    (NOC coordinates; core t is the root of row t) and 0 / 1 selection tiles per row count."""

    def __init__(self, pk, x0, y0, CX, CY):
        self.pk, self.x0, self.y0, self.CX, self.CY, self.C = pk, x0, y0, CX, CY, CX * CY
        assert pk.Ht % self.C == 0
        self.PER = pk.Ht // self.C
        self.cores = [ttnn.CoreCoord(x0 + i % CX, y0 + i // CX) for i in range(self.C)]
        self.crs = _range_set(self.cores)
        lo = pk.mesh.worker_core_from_logical_core(ttnn.CoreCoord(x0, y0))
        hi = pk.mesh.worker_core_from_logical_core(ttnn.CoreCoord(x0 + CX - 1, y0 + CY - 1))
        self.mc = [lo.x, lo.y, hi.x, hi.y]
        self.roots = [v for cc in self.cores[:16] for w in [pk.mesh.worker_core_from_logical_core(cc)] for v in (w.x, w.y)]
        self.reds = {}

    def red(self, T):
        K = red_k(T)
        if K not in self.reds:
            self.reds[K] = red_sel(self.pk.mesh, self.pk.rep, self.C, K)
        return self.reds[K]


class GateKernel2:
    """The message gate on a core rectangle with its rounds (tt/kernels/gate2_*.cpp): live rows in / out, round-1 norm
    statistics and round-2 partial down projections reduced exactly by 0 / 1 matmuls at the root (hi / lo halves), the
    totals multicast back. <= 16 rows. With nw: the following RMS norm too (round 3; bf16 out); with router (W hi, W lo)
    bf16 [1, 1, H, E] (<= 8 rows): the router logits (round 4) -> [1, 1, max(T, 2), E] fp32. dup (T = 1): row 0 also
    into row 1 of the outputs (the MoE's two-row path). multi(): up to three gates in one program, each on its own
    rectangle (columns 0..8: rows 0..3, 4..7, 8..9), their rounds in parallel."""

    ROUTER_ROWS = 8  # (the gather buffer: 2 E / 32 parts of ceil(C / 4 red_k) tiles, a root per row)

    def __init__(self, post, R: int):
        from .post import CX, CY

        self.pk, self.R = post, R // 32
        self.H, self.Ht, self.eps = post.H, post.Ht, post.eps
        self.rects = [_Rect(post, 0, 0, CX, CY)]
        g = post.mesh.compute_with_storage_grid_size()
        if CX == 9 and CY == 4 and g.y >= 10:
            self.rects += [_Rect(post, 0, 4, 9, 4), _Rect(post, 0, 8, 9, 2)]

    def _parts(self, rc, h, m, wd, wfg, nw, router, dup, out_mc):
        pk = self.pk
        h, m = (ttnn.sharded_to_interleaved(t, ttnn.L1_MEMORY_CONFIG) if t.is_sharded() else t for t in (h, m))
        T = int(h.shape[-2])
        norm, rt = nw is not None, router is not None
        assert T <= 16 and h.dtype == ttnn.float32 and m.dtype == ttnn.float32 and (norm or not rt)
        assert not rt or T <= self.ROUTER_ROWS
        dup = dup and T == 1
        odt = ttnn.bfloat16 if norm else ttnn.float32
        out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(h.shape)), odt, ttnn.TILE_LAYOUT, pk.mesh, out_mc)
        ET = int(router[0].shape[-1]) // 32 if rt else 4
        lg = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, max(T, 2), 32 * ET]), ttnn.float32, ttnn.TILE_LAYOUT,
                                            pk.mesh, out_mc) if rt else None
        NSEL, sel = rc.red(T)  # (a root per row)
        rsplit = RSPLIT and rt and T * ET <= 16
        PER, R, C, crs = rc.PER, self.R, rc.C, rc.crs
        GP = 2 * ET if rt else 4
        (cb_h, cb_m, cb_wd, cb_wf, cb_wg, cb_ones, cb_sel, cb_sq, cb_part, cb_gath, cb_tout, cb_tot, cb_r, cb_hn,
         cb_tot2, cb_g, cb_fa, cb_out, cb_zero, cb_o32, cb_tot3, cb_nw, cb_xb, cb_rw) = range(24)
        from .post import cbt, tile_rows

        f32, bf, th = ttnn.float32, ttnn.bfloat16, tile_rows(T)
        r = lambda i, dt, n: cbt(crs, i, dt, n, th)  # noqa: E731  (token rows: th x 32 tiles)
        nparts = 8 + (2 if norm else 0) + (2 * ET if rt else 0)  # (every round's parts contiguous: no wrap inside one)
        ntout = 4 + (1 if norm else 0) + (ET if rt else 0)
        cbs = [r(cb_h, f32, PER), r(cb_m, f32, PER), _cb(crs, cb_wd, bf, BF, 2 * PER * R),
               _cb(crs, cb_wf, bf, BF, R * PER), _cb(crs, cb_wg, bf, BF, R * PER), _cb(crs, cb_ones, bf, BF, 1),
               r(cb_sel, bf, NSEL), r(cb_sq, f32, 4 * PER), r(cb_part, f32, nparts),
               _cb(crs, cb_gath, f32, F32, GP * NSEL), r(cb_tout, f32, ntout),
               r(cb_tot, f32, 2), r(cb_r, f32, 2), r(cb_hn, bf, 2 * PER),
               r(cb_tot2, f32, 2), r(cb_g, bf, R), r(cb_fa, f32, 2 * PER),
               r(cb_out, odt, 2), _cb(crs, cb_zero, bf, BF, 1)]
        if norm:
            cbs += [r(cb_o32, f32, PER), r(cb_tot3, f32, 1), r(cb_nw, f32, PER)]
        if rt:
            cbs += [r(cb_xb, bf, PER), _cb(crs, cb_rw, bf, BF, 2 * PER * ET)]
        rhi, rlo = router if rt else (wd, wd)
        rd_ct = [cb_h, cb_m, cb_wd, cb_wf, cb_wg, cb_ones, cb_sel, PER, R, self.Ht, NSEL, cb_nw, cb_rw, int(norm), int(rt),
                 ET] + ttnn.TensorAccessorArgs(h).get_compile_time_args() \
            + ttnn.TensorAccessorArgs(wd).get_compile_time_args() + ttnn.TensorAccessorArgs(m).get_compile_time_args()
        rd_common = [h.buffer_address(), m.buffer_address(), wd.buffer_address(), wfg.buffer_address(),
                     pk.ones.buffer_address(), sel.buffer_address(), (nw if norm else wd).buffer_address(),
                     rhi.buffer_address(), rlo.buffer_address()]
        wr_ct = [cb_part, cb_gath, cb_tout, cb_tot, cb_tot2, cb_tot3, cb_out, cb_zero, PER, C, NSEL, GP, int(norm),
                 int(rt), ET, int(dup)] + ttnn.TensorAccessorArgs(out).get_compile_time_args() \
            + ttnn.TensorAccessorArgs(lg if rt else out).get_compile_time_args() \
            + ttnn.TensorAccessorArgs(pk.zeros).get_compile_time_args()
        wr_common = [out.buffer_address(), (lg if rt else out).buffer_address()] + rc.mc + [pk.zeros.buffer_address()] \
            + rc.roots[:2 * (T * ET if rsplit else T)]
        cp_ct = [cb_h, cb_m, cb_wd, cb_wf, cb_wg, cb_ones, cb_sel, cb_sq, cb_part, cb_gath, cb_tout, cb_tot, cb_r, cb_hn,
                 cb_gath, cb_tot2, cb_g, cb_fa, cb_out, PER, R, NSEL, _bits(1.0 / self.H), _bits(self.eps), GP,
                 int(norm), int(rt), ET, cb_o32, cb_tot3, cb_nw, cb_xb, cb_rw]
        u2d = [ttnn.UnpackToDestMode.Default] * 64
        for i in (cb_h, cb_m, cb_tot, cb_r, cb_tot2, cb_fa, cb_o32, cb_tot3, cb_nw):
            u2d[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           dst_full_sync_en=True, math_approx_mode=False)
        cfg.unpack_to_dest_mode = u2d
        defs = [("G2_RK", str(red_k(T))), ("G2_CX", str(rc.CX)), ("G2_ROWS", str(T)), ("G2_X0", str(rc.x0)), ("G2_Y0", str(rc.y0)),
                ("G2_TH", str(th)), ("G2_ZLAZY", str(int(ZLAZY))), ("G2_RSPLIT", str(int(rsplit)))]

        def dm(src, ct, common, riscv, noc):
            kd = ttnn.KernelDescriptor(kernel_source=str(KDIR / src), core_ranges=crs, compile_time_args=ct,
                                       defines=defs, config=ttnn.DataMovementConfigDescriptor(processor=riscv, noc=noc))
            kd.common_runtime_args = common
            return kd

        kernels = [
            dm("gate2_reader.cpp", rd_ct, rd_common, ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1),
            dm("gate2_writer.cpp", wr_ct, wr_common, ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "gate2_compute.cpp"), core_ranges=crs, compile_time_args=cp_ct,
                                  runtime_args=_rt(rc.cores, [[i] for i in range(C)]), config=cfg,
                                  defines=[("ROWS", str(T)), ("G2_RSPLIT", str(int(rsplit)))]),
        ]
        key = (T, th, rc.x0, rc.y0, rc.CX, rc.CY, norm, rt, ET, dup, ZLAZY, rsplit,
               tuple(repr(t.memory_config()) for t in (h, m, wd, out) + ((lg,) if rt else ())))
        io = [h, m, wd, wfg, pk.ones, sel, pk.zeros] + ([nw] if norm else []) + ([rhi, rlo] if rt else []) + [out] \
            + ([lg] if rt else [])
        return kernels, cbs, io, key, ((out, lg) if norm else out)

    def _run(self, parts, crs):
        kernels, cbs, io, keys, res = [], [], [], [], []
        for k, c, i, key, r in parts:
            kernels += k
            cbs += c
            io += i
            keys.append(key)
            res.append(r)
        sems = [ttnn.SemaphoreDescriptor(i, ttnn.CoreType.WORKER, crs, 0) for i in range(5)]
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        prog.custom_program_hash = int.from_bytes(hashlib.sha1(repr(("i2m_gate2", self.R, keys)).encode()).digest()[:8],
                                                  "little") & ((1 << 63) - 1)
        uio, seen = [], set()
        for t in io:
            if id(t) not in seen:
                seen.add(id(t))
                uio.append(t)
        ttnn.generic_op(uio, prog)
        return res

    def __call__(self, h, m, wd, wfg, nw=None, router=None, dup=False, out_mc=ttnn.L1_MEMORY_CONFIG):
        rc = self.rects[0]
        return self._run([self._parts(rc, h, m, wd, wfg, nw, router, dup, out_mc)], rc.crs)[0]

    def multi(self, gates, out_mc=ttnn.L1_MEMORY_CONFIG):
        """gates: up to len(self.rects) (h, m, wd, wfg, nw) -> their outputs ((x, None) with nw), one program."""
        assert len(gates) <= len(self.rects)
        rects = self.rects[: len(gates)]
        parts = [self._parts(rc, h, m, wd, wfg, nw, None, False, out_mc) for rc, (h, m, wd, wfg, nw) in zip(rects, gates)]
        crs = _range_set([c for rc in rects for c in rc.cores])
        return self._run(parts, crs)
