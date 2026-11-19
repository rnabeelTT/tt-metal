# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused hyper-connection for decode passes (one 32-row tile row; X [1, 4, 32, H] bf16, L1 interleaved).

One generic_op per sublayer boundary instead of ~20 ttnn ops per HC pre and ~5 per HC post (tt/hc.py):

    POST: X' = bf16(X + post (x) o)          (o: the previous sublayer's normed output, post from its HC's weights)
    PRE:  w  = aff * sigmoid(((X' fn^T) * rsqrt(mean(X'^2) + eps)) * scale + base) + hc_eps ;  in = bf16(sum_s pre_s X'_s)

NC = 16 cores, each owning H / 32 / NC hidden tiles of the 4 streams; the per-core partial mix (X' @ fn^T) and sum of
squares (diag of X' X'^T) are exchanged between the cores (the qwen36 addnorm pattern) and every core finishes the
weights in FP32. Numerics: X' and the sums of squares exact, fn / pre / post weights at the FPU's tf32 operand width
(as ttnn's fp32 matmul of the unfused path), everything else FP32. kernels/hc_*.cpp.
"""

from __future__ import annotations

import os as _os
import struct
from pathlib import Path

import torch
import ttnn

KDIR = Path(__file__).parent / "kernels"
TILE = 32
SEM, SEM_READY = 0, 1


def _bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _cb(cores, index, dtype, tile_bytes, tiles):
    fmt = ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=tile_bytes)
    return ttnn.CBDescriptor(total_size=tile_bytes * tiles, core_ranges=cores, format_descriptors=[fmt])


class HCFusedOp:
    """Model-wide program builder (cores, NOC coordinates, the shared matmul constants)."""

    _OPS = {}

    @classmethod
    def get(cls, mesh, hidden, nc=None):
        nc = nc or int(_os.environ.get("HY4_HCF_NC", "16"))
        key = (id(mesh), hidden, nc)
        if key not in cls._OPS:
            cls._OPS[key] = cls(mesh, hidden, nc)
        return cls._OPS[key]

    def __init__(self, mesh, hidden, nc=16):
        from models.demos.blackhole.qwen36.tt.moe_decode.experts import noc_coords

        self.mesh, self.H, self.NC = mesh, hidden, nc
        self.HT = hidden // TILE
        assert self.HT % nc == 0
        self.J = self.HT // nc
        self.cores = [ttnn.CoreCoord(i % 8, i // 8) for i in range(nc)]
        self.crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, nc // 8 - 1))])
        coords = noc_coords(mesh)
        self.noc = [v for c in self.cores for v in coords[(c.x, c.y)]]
        self.rep = {"mesh_mapper": ttnn.ReplicateTensorToMesh(mesh)} if mesh.get_num_devices() > 1 else {}
        mm = torch.zeros(10, TILE, TILE)
        mm[0] = 1.0
        for m in range(8):
            mm[1 + m, m, :] = 1.0  # w @ sel_m: every column = w[:, m]
        mm[9] = torch.eye(TILE)  # hi = W @ I: W truncated to tf32 (the hi / lo splits)
        self.mm = self.upload(mm.reshape(1, 1, 10 * TILE, TILE))

    def upload(self, t, dtype=ttnn.float32):
        return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG, **self.rep)

    def __call__(self, X, o=None, wp=None, hc=None):
        """X [1, 4, 32, H] bf16; o [1, 1, rows, H] bf16 + wp [1, 1, 32, 32] fp32 (POST, else None); hc: the next
        HyperConnection, with its fused constants (PRE, else None). -> (X', in, w): X' (X itself when no POST), in / w
        (None when no PRE)."""
        post, pre = o is not None, hc is not None
        J, NC, H = self.J, self.NC, self.H
        NT = 4 * J
        L1 = ttnn.L1_MEMORY_CONFIG
        alloc = lambda shape, dt: ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT,  # noqa: E731
                                                                  self.mesh, L1)
        # the residual streams live in DRAM (768 KB: in L1 the two copies alive across a call crowded the streamed
        # linears' circular buffers off the cores); in / w (one tile row / one tile) in L1
        xn = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 4, 32, H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh,
                                            ttnn.DRAM_MEMORY_CONFIG) if post else X
        out_mem = L1 if _os.environ.get("HY4_HCF_OUT_L1", "0") == "1" else ttnn.DRAM_MEMORY_CONFIG
        inp = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32, H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh,
                                             out_mem) if pre else None
        w = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32, 32]), ttnn.float32, ttnn.TILE_LAYOUT, self.mesh,
                                           out_mem) if pre else None
        o_t = o if post else X
        wp_t = wp if post else self.mm
        fn_t = hc.fz_fn if pre else self.mm
        fl_t = hc.fz_fnlo if pre else self.mm
        cv_t = hc.fz_cv if pre else self.mm
        bf, f32 = (ttnn.bfloat16, 2048), (ttnn.float32, 4096)
        cbs = [_cb(self.crs, i, dt, tb, n) for i, (dt, tb), n in [
            (0, bf, NT), (1, bf, J), (2, f32, 1), (3, f32, NT), (4, f32, 10), (5, f32, 4), (6, f32, 4), (7, bf, NT),
            (8, f32, 1), (9, f32, 2), (10, f32, 2 * NC), (11, f32, 1), (12, f32, 1), (13, f32, 4), (14, f32, 1),
            (16, bf, J), (17, f32, 1), (18, f32, 1), (19, f32, 1), (20, f32, 1), (21, f32, 1), (22, bf, NT),
            (23, f32, 4), (24, f32, 4)]]
        acc = lambda t: ttnn.TensorAccessorArgs(t).get_compile_time_args()  # noqa: E731
        fnlo = int(pre and _os.environ.get("HY4_HCF_FNLO", "0") == "1")
        rd_ct = [J, self.HT, int(post), int(pre), 0, 1, 2, 3, 4, 5, 10, 4, 17, 22, fnlo]
        rd_ct += acc(X) + acc(o_t) + acc(wp_t) + acc(fn_t) + acc(self.mm) + acc(cv_t) + acc(fl_t)
        rd_rt = [[X.buffer_address(), o_t.buffer_address(), wp_t.buffer_address(), fn_t.buffer_address(),
                  self.mm.buffer_address(), cv_t.buffer_address(), c, fl_t.buffer_address()] for c in range(NC)]
        out_in, out_w = (inp, w) if pre else (xn, xn)
        wr_ct = [J, self.HT, int(post), int(pre), NC, SEM, SEM_READY, 9, 10, 7, 16, 11]
        wr_ct += acc(xn) + acc(out_in) + acc(out_w)
        wr_rt = [[xn.buffer_address(), out_in.buffer_address(), out_w.buffer_address(), c] + self.noc
                 for c in range(NC)]
        n_hc = hc.hc * hc.H if pre else 4 * H

        cp_ct = [J, int(post), int(pre), NC, _bits(hc.eps if pre else 1e-6), _bits(1.0 / n_hc),
                 _bits(hc.hc_eps if pre else 0.0), int(_os.environ.get("HY4_HCF_DBG", "0")), fnlo]
        full = _os.environ.get("HY4_HCF_DST_FULL", "1") == "1"  # 8 FP32 DST tiles per round (fewer op inits)
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           math_approx_mode=False, dst_full_sync_en=full)
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for i in (5, 14, 17, 18, 21):  # cv, M, wp / w / S*I copies: FP32 copies into DST (slots: FPU operands)
            modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = modes

        def rt(args):
            r = ttnn.RuntimeArgs()
            for c, a in zip(self.cores, args):
                r[c.x][c.y] = a
            return r

        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "hc_reader.cpp"), core_ranges=self.crs,
                                  compile_time_args=rd_ct, runtime_args=rt(rd_rt),
                                  config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1)),
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "hc_writer.cpp"), core_ranges=self.crs,
                                  compile_time_args=wr_ct, runtime_args=rt(wr_rt),
                                  config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0)),
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "hc_compute.cpp"), core_ranges=self.crs,
                                  compile_time_args=cp_ct, runtime_args=rt([[] for _ in range(NC)]), config=cfg,
                                  defines=[("DST_FULL", "1")] if full else []),
        ]
        sems = [ttnn.SemaphoreDescriptor(id=i, core_type=ttnn.CoreType.WORKER, core_ranges=self.crs, initial_value=0)
                for i in (SEM, SEM_READY)]
        io = [X, o_t, wp_t, fn_t, fl_t, self.mm, cv_t] + ([xn] if post else []) + ([inp, w] if pre else [])
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs))
        return xn, inp, w


def fused_consts(op, hcobj, fn, base, scale):
    """The HC's fused-kernel constants: fz_fn [1, 1, hc H, 32] fp32 (fn^T, stream-major rows, M columns used),
    fz_cv [1, 1, 4 * 32, 32] fp32 (scale / base / affine rows repeated over 32 rows, identity)."""
    hc, H, M = hcobj.hc, hcobj.H, hcobj.M
    fnT = torch.zeros(hc * H, TILE)
    fnT[:, :M] = fn.float().T
    sc = torch.zeros(TILE)
    bs = torch.zeros(TILE)
    a = torch.zeros(TILE)
    scale = scale.float().reshape(-1)
    if M == 2 * hc:
        sc[:hc], sc[hc:M] = scale[0], scale[1]
        a[:hc], a[hc:M] = 1.0, float(hcobj.mag)
    else:
        sc[:M] = scale[0]
        a[:M] = 1.0
    bs[:M] = base.float().reshape(-1)
    cv = torch.stack([sc.expand(TILE, TILE), bs.expand(TILE, TILE), a.expand(TILE, TILE), torch.eye(TILE)])
    # fn = hi + lo: hi exactly representable at the FPU's tf32 operand width (mantissa cut to 10 bits), lo the rest in
    # bf16 (~2^-19 of fn): the partial mix X' @ hi + X' @ lo is then ~fp32 instead of tf32
    hi = (fnT.contiguous().view(torch.int32) & ~0x1FFF).view(torch.float32)
    hcobj.fz_fn = op.upload(hi.reshape(1, 1, hc * H, TILE))
    hcobj.fz_fnlo = op.upload((fnT - hi).reshape(1, 1, hc * H, TILE), ttnn.bfloat16)
    hcobj.fz_cv = op.upload(cv.reshape(1, 1, 4 * TILE, TILE))
