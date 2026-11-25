# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill RMS norm over whole tile rows (tt/kernels/rown_*.cpp): ttnn.rms_norm's arithmetic (layernorm.cpp, RMSNORM,
fp32 DST, HiFi4) with the input row read once and kept in L1, a bf16 input (exact: what ttnn.typecast to fp32 then
rms_norm see) and the bf16 output cast (ttnn.typecast's SFPU cast) in the same program; the output may go to several
destinations, each a column range of a wider tensor (a message gate's [rms(h) | rms(m)] input: no concat).
Bitwise equal to [typecast ->] ttnn.rms_norm [-> typecast]. One core per row tile (rows over the grid)."""

from __future__ import annotations

import hashlib
import struct

import ttnn

from .bconv import KDIR, _cb, _range_set, _rt
from .bconv import _ks

F32, BF = 4096, 2048
BLK = 4


class RowNorm:
    def __init__(self, mesh, H: int, eps: float):
        self.mesh, self.Wt, self.eps = mesh, H // 32, eps
        assert self.Wt % BLK == 0
        grid = mesh.compute_with_storage_grid_size()
        self.G, self.gx = grid.x * grid.y, grid.x
        self.progs = {}
        self.grows = {}  # id(weight) -> (weight, its row-major bf16 row)
        self.eps_bits = struct.unpack("<I", struct.pack("<f", eps))[0]

    def _grow(self, w):
        """the weight w [1, 1, 1, H] fp32 (the checkpoint's bf16 values) as one row-major bf16 row (4.5 KB, cached)"""
        if id(w) not in self.grows:
            import torch

            r = ttnn.to_torch(ttnn.get_device_tensors(w)[0] if self.mesh.get_num_devices() > 1 else w).float()
            r = r.reshape(-1)[: 32 * self.Wt]
            assert torch.equal(r, r.to(torch.bfloat16).float())
            self.grows[id(w)] = (w, ttnn.from_torch(r.to(torch.bfloat16).reshape(1, 1, 1, -1), dtype=ttnn.bfloat16,
                                                    layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh,
                                                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)
                                                    if self.mesh.get_num_devices() > 1 else None))
        return self.grows[id(w)][1]

    def ok(self, x):
        return (len(x.shape) == 4 and int(x.shape[0]) * int(x.shape[1]) == 1 and int(x.shape[2]) % 32 == 0
                and int(x.shape[3]) == 32 * self.Wt and x.layout == ttnn.TILE_LAYOUT and not x.is_sharded()
                and x.dtype in (ttnn.float32, ttnn.bfloat16))

    def __call__(self, x, w=None, out_dtype=ttnn.float32, dests=None, out_mc=ttnn.DRAM_MEMORY_CONFIG, qrep=None):
        """x [1, 1, T, H] fp32 / bf16 tile -> rms(x) [* w] (w [1, 1, 1|32, H] fp32) as out_dtype, [1, 1, T, H] (returned),
        or into dests: [(tensor [1, 1, T, k H], column tile offset)] (all of out_dtype, one memory config).
        qrep ([1, 1, H, nq * 32] fp32, no w, fp32 output): -> (out, out @ qrep [1, 1, T, nq * 32] fp32) (ROWN_LG, as
        ttnn.matmul(out, qrep) computes it)."""
        return self.multi([dict(x=x, w=w, out_dtype=out_dtype, dests=dests, qrep=qrep)], out_mc)[0]

    def multi(self, jobs, out_mc=ttnn.DRAM_MEMORY_CONFIG):
        """several norms (dicts of __call__'s arguments, the same rows) in ONE program, each on its own cores ->
        their results (a prefill chunk's few-hundred-row norms are latency-bound on one core per row tile)"""
        Wt = self.Wt
        res, keys, prep = [], [], []
        for jb in jobs:
            x, qrep, out_dtype = jb["x"], jb.get("qrep"), jb.get("out_dtype", ttnn.float32)
            T = int(x.shape[2])
            dests, out = jb.get("dests"), None
            if dests is None:
                out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, 32 * Wt]), out_dtype, ttnn.TILE_LAYOUT,
                                                     self.mesh, out_mc)
                dests = [(out, 0)]
            g = self._grow(jb["w"]) if jb.get("w") is not None else None
            nq = int(qrep.shape[-1]) // 32 if qrep is not None else 0
            assert qrep is None or (g is None and out_dtype == ttnn.float32)
            lg = None
            if qrep is not None:
                lg = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, 32 * nq]), ttnn.float32, ttnn.TILE_LAYOUT,
                                                    self.mesh, ttnn.DRAM_MEMORY_CONFIG)
            d0 = dests[0][0]
            keys.append((T, x.dtype, out_dtype, g is not None, len(dests), nq, x.memory_config().buffer_type,
                         d0.memory_config().buffer_type))
            prep.append((x, g, qrep, dests, lg))
            res.append(out if qrep is None else (out, lg))
        key = tuple(keys)
        if key not in self.progs:
            n = len(jobs)
            parts, c0 = [], 0
            for i, (k, (x, g, qrep, dests, lg)) in enumerate(zip(keys, prep)):
                assert all(d.dtype == k[2] for d, _ in dests)
                ncores = min(k[0] // 32, (self.G - c0) // (n - i))
                parts.append(self._build(k, x, g, qrep, dests[0][0], lg, c0, ncores))
                c0 += ncores
            hk = ("i2m_rown5", repr(key), tuple(p[-1] for p in parts))
            h = int.from_bytes(hashlib.sha1(repr(hk).encode()).digest()[:8], "little") & ((1 << 63) - 1)
            self.progs[key] = ([p[:4] for p in parts], h)
        parts, h = self.progs[key]
        kernels, cbs, io = [], [], []
        for (rd, wr, cm, cb), (x, g, qrep, dests, lg) in zip(parts, prep):
            rd.common_runtime_args = [x.buffer_address(), g.buffer_address() if g is not None else 0, self.eps_bits,
                                      qrep.buffer_address() if qrep is not None else 0]
            wr.common_runtime_args = [len(dests)] + [v for d, off in dests
                                                     for v in (d.buffer_address(), int(d.shape[-1]) // 32, off)] \
                + ([lg.buffer_address()] if lg is not None else [])
            kernels += [rd, wr, cm]
            cbs += cb
            io += [x] + ([g] if g is not None else []) + ([qrep, lg] if qrep is not None else []) + [d for d, _ in dests]
        uniq, seen = [], set()
        for t in io:
            if id(t) not in seen:
                seen.add(id(t))
                uniq.append(t)
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        prog.custom_program_hash = h
        ttnn.generic_op(uniq, prog)
        return res

    def _build(self, key, x, g, qrep, d0, lg, c0, nc):
        """one norm's kernels and CBs on cores c0 .. c0 + nc - 1 (per call only the common runtime args change: the
        tensors' addresses)"""
        T, _, out_dtype, _, _, nq, _, _ = key
        Wt = self.Wt
        RT = T // 32
        work = [(RT * i // nc, RT * (i + 1) // nc - RT * i // nc) for i in range(nc)]
        cores = [ttnn.CoreCoord(i % self.gx, i // self.gx) for i in range(c0, c0 + nc)]
        crs = _range_set(cores)
        xb = F32 if x.dtype == ttnn.float32 else BF
        ob = F32 if out_dtype == ttnn.float32 else BF
        cb_in, cb_sq, cb_acc, cb_sc, cb_eps, cb_ex2, cb_ex2pe, cb_fus, cb_g, cb_out = 0, 1, 2, 3, 4, 5, 6, 7, 8, 16
        cb_y, cb_q, cb_lacc, cb_lg = 9, 10, 11, 12
        f32, bf = ttnn.float32, ttnn.bfloat16
        cbs = [_cb(crs, cb_in, x.dtype, xb, Wt), _cb(crs, cb_sq, f32, F32, 2 * BLK), _cb(crs, cb_acc, f32, F32, 1),
               _cb(crs, cb_sc, bf, BF, 1), _cb(crs, cb_eps, bf, BF, 1), _cb(crs, cb_ex2, f32, F32, 1),
               _cb(crs, cb_ex2pe, f32, F32, 1), _cb(crs, cb_out, out_dtype, ob, 2 * BLK)]
        defs = []
        if g is not None:
            cbs += [_cb(crs, cb_fus, f32, F32, BLK), _cb(crs, cb_g, bf, BF, Wt)]
            defs.append(("ROWN_GAMMA", "1"))
        if out_dtype == bf:
            defs.append(("ROWN_BF16", "1"))
        if qrep is not None:
            defs.append(("ROWN_LG", "1"))
            cbs += [_cb(crs, cb_y, f32, F32, BLK), _cb(crs, cb_q, f32, F32, 2 * BLK * nq),
                    _cb(crs, cb_lacc, f32, F32, nq), _cb(crs, cb_lg, f32, F32, nq)]
        inv_w = struct.unpack("<I", struct.pack("<f", 1.0 / (32 * Wt)))[0]
        rd_ct = [cb_in, cb_sc, cb_eps, cb_g, Wt, xb, cb_q, max(nq, 1), cb_fus] + ttnn.TensorAccessorArgs(x).get_compile_time_args()
        rd_ct += ttnn.TensorAccessorArgs(g if g is not None else x).get_compile_time_args()
        if qrep is not None:
            rd_ct += ttnn.TensorAccessorArgs(qrep).get_compile_time_args()
        wr_ct = [cb_out, Wt, ob] + ttnn.TensorAccessorArgs(d0).get_compile_time_args()
        if qrep is not None:
            wr_ct += [cb_lg, nq] + ttnn.TensorAccessorArgs(lg).get_compile_time_args()
        u2d = [ttnn.UnpackToDestMode.Default] * 64
        u2d[cb_acc] = ttnn.UnpackToDestMode.UnpackToDestFp32
        u2d[cb_lacc] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           dst_full_sync_en=False, math_approx_mode=False)
        cfg.unpack_to_dest_mode = u2d
        cm = [cb_in, cb_sq, cb_acc, cb_sc, cb_eps, cb_ex2, cb_ex2pe, cb_fus, cb_g, cb_out, Wt, inv_w,
              cb_y, cb_q, cb_lacc, cb_lg, max(nq, 1)]
        per_core = [[r0, n] for r0, n in work]
        dm = lambda src, ct, riscv, noc: ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=_ks(KDIR, src), core_ranges=crs, compile_time_args=ct, runtime_args=_rt(cores, per_core),
            config=ttnn.DataMovementConfigDescriptor(processor=riscv, noc=noc), defines=defs)
        rd = dm("rown_reader.cpp", rd_ct, ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0)
        wr = dm("rown_writer.cpp", wr_ct, ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1)
        comp = ttnn.KernelDescriptor(kernel_source=_ks(KDIR, "rown_compute.cpp"), core_ranges=crs, compile_time_args=cm,
                                     runtime_args=_rt(cores, [[n] for _, n in work]), config=cfg, defines=defs)
        return rd, wr, comp, cbs, (c0, nc, repr(x.memory_config()), repr(d0.memory_config()))
