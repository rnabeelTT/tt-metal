# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One-program batched short conv step (tt/kernels/bconv_*.cpp) for the decode of B slots (one row each):

    y = x * (w3 + 1) + s0 * w0 + s1 * w1 + s2 * w2        (fp32 in DST, y bf16)
    s0, s1, s2 <- s1, s2, x                               (in place, fp32)

Replaces ~11 ttnn ops per conv (4 convs per layer application). States [1, 1, B, C] fp32 TILE (B <= 32: one tile row),
taps [1, 1, 32, C] fp32 TILE (every row = the tap), channel tiles split over the cores."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

import ttnn

KDIR = Path(__file__).parent / "kernels"
# I2M_BCONV_ROWS: <= 16 live rows -> the conv's SFPU work on those rows only (bconv_compute.cpp BCONV_ROWS)
ROWS = __import__("os").environ.get("I2M_BCONV_ROWS", "1") == "1"
TILE = 32


def _range_set(cores):
    rs = ttnn.CoreRangeSet([ttnn.CoreRange(cores[0], cores[0])])
    for c in cores[1:]:
        rs = rs.merge(ttnn.CoreRangeSet([ttnn.CoreRange(c, c)]))
    return rs


def _cb(cores, index, dtype, tile_bytes, tiles):
    fmt = ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=tile_bytes)
    return ttnn.CBDescriptor(total_size=tile_bytes * tiles, core_ranges=cores, format_descriptors=[fmt])


def _rt(cores, args):
    rt = ttnn.RuntimeArgs()
    for c, a in zip(cores, args):
        rt[c.x][c.y] = a
    return rt


class BConvKernel:
    def __init__(self, mesh, taps: torch.Tensor, mapper, c_local: int):
        """taps: [C_full, 4] (the checkpoint's conv weight); mapper shards the channel dim like the activations."""
        self.mesh, self.C = mesh, c_local
        up = lambda t: ttnn.from_torch(t.reshape(1, 1, 1, -1).expand(1, 1, TILE, -1).contiguous(),  # noqa: E731
                                       dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh,
                                       memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)
        self.w = [up(taps[:, j].float()) for j in range(3)]
        self.wc = up(taps[:, 3].float() + 1.0)
        self.Ct = c_local // TILE
        grid = mesh.compute_with_storage_grid_size()
        ncores = min(self.Ct, grid.x * grid.y)
        while self.Ct % ncores:
            ncores -= 1
        self.cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(ncores)]
        self.per = self.Ct // ncores
        self.crs = _range_set(self.cores)

    def step(self, x, s, out_mc=ttnn.L1_MEMORY_CONFIG, taps=None):
        kernels, cbs, io, key, y = self.parts(x, s, out_mc, taps)
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        prog.custom_program_hash = _hash(key)
        ttnn.generic_op(io, prog)
        return y

    def parts(self, x, s, out_mc=ttnn.L1_MEMORY_CONFIG, taps=None, cores=None):
        """cores: this conv's cores (default its own). x [1, 1, B, C] TILE interleaved (bf16 or fp32), s = [s0, s1, s2] fp32 TILE (updated in place) -> y bf16.
        taps: (w0, w1, w2, wc) [1, 1, B, C] fp32 per-row taps (the dynamic convs) instead of the static ones, or
        (cat,): one [1, 1, B, 4 C] tensor [w0 | w1 | w2 | wc]."""
        cores = self.cores if cores is None else cores
        crs = _range_set(cores)
        stride = 0
        if taps is not None and len(taps) == 1:
            stride = int(x.shape[-1]) // 32
            taps = (taps[0],) * 4
        w_taps, wc = (list(taps[:3]), taps[3]) if taps is not None else (self.w, self.wc)
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        y = ttnn.allocate_tensor_on_device(ttnn.Shape(list(x.shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh,
                                           out_mc)
        xb = 4096 if x.dtype == ttnn.float32 else 2048
        cb_x, cb_s, cb_w, cb_y, cb_o0, cb_o1, cb_o2 = 0, 1, 2, 3, 4, 5, 6
        f32, bf = ttnn.float32, ttnn.bfloat16
        cbs = [_cb(crs, cb_x, x.dtype, xb, 2), _cb(crs, cb_s, f32, 4096, 3), _cb(crs, cb_w, f32, 4096, 4),
               _cb(crs, cb_y, bf, 2048, 2)] + [_cb(crs, i, f32, 4096, 2) for i in (cb_o0, cb_o1, cb_o2)]
        rd_ct = [cb_x, cb_s, cb_w, xb] + ttnn.TensorAccessorArgs(x).get_compile_time_args() \
            + ttnn.TensorAccessorArgs(s[0]).get_compile_time_args() + ttnn.TensorAccessorArgs(w_taps[0]).get_compile_time_args()
        wr_ct = [cb_y, cb_o0, cb_o1, cb_o2] + ttnn.TensorAccessorArgs(y).get_compile_time_args() \
            + ttnn.TensorAccessorArgs(s[0]).get_compile_time_args()
        addr = [x.buffer_address()] + [t.buffer_address() for t in s] + [t.buffer_address() for t in w_taps] \
            + [wc.buffer_address()]
        rd_rt = [addr + [i * self.per, self.per] for i in range(len(cores))]
        wr_rt = [[y.buffer_address()] + [t.buffer_address() for t in s] + [i * self.per, self.per]
                 for i in range(len(cores))]
        u2d = [ttnn.UnpackToDestMode.Default] * 64  # (this build: one entry per CB index, 64)
        for i in (cb_s, cb_w) + ((cb_x,) if x.dtype == ttnn.float32 else ()):
            u2d[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           dst_full_sync_en=True, math_approx_mode=False)
        cfg.unpack_to_dest_mode = u2d
        dm = lambda src, ct, rt, riscv, noc: ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=str(KDIR / src), core_ranges=crs, compile_time_args=ct, runtime_args=_rt(cores, rt),
            defines=[("BCONV_TAP_STRIDE", str(stride))] if stride else [],
            config=ttnn.DataMovementConfigDescriptor(processor=riscv, noc=noc))
        kernels = [
            dm("bconv_reader.cpp", rd_ct, rd_rt, ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
            dm("bconv_writer.cpp", wr_ct, wr_rt, ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
            ttnn.KernelDescriptor(kernel_source=str(KDIR / "bconv_compute.cpp"), core_ranges=crs,
                                  compile_time_args=[cb_x, cb_s, cb_w, cb_y, cb_o0, cb_o1, cb_o2],
                                  runtime_args=_rt(cores, [[self.per]] * len(cores)), config=cfg,
                                  defines=[("BCONV_ROWS", str(int(x.shape[-2])))] if ROWS and int(x.shape[-2]) <= 16
                                  else []),
        ]
        key = ("i2m_bconv", self.C, len(cores), cores[0].x, cores[0].y, str(x.dtype), int(x.shape[-2]), stride,
               repr(x.memory_config()), repr(out_mc), repr(s[0].memory_config()), repr(w_taps[0].memory_config()))
        return kernels, cbs, [x] + list(s) + list(w_taps) + [wc, y], key, y


def _hash(key):
    return int.from_bytes(hashlib.sha1(repr(key).encode()).digest()[:8], "little") & ((1 << 63) - 1)


def bconv_pair(ka: "BConvKernel", xa, sa, kb: "BConvKernel", xb, sb, out_mc=ttnn.L1_MEMORY_CONFIG):
    """Two static convs (a sliding layer's k and v) in ONE program: b on the cores after a's -> (ya, yb)."""
    grid = ka.mesh.compute_with_storage_grid_size()
    n = len(ka.cores)
    assert n + len(kb.cores) <= grid.x * grid.y
    cb_ = [ttnn.CoreCoord((n + i) % grid.x, (n + i) // grid.x) for i in range(len(kb.cores))]
    k1, c1, io1, key1, ya = ka.parts(xa, sa, out_mc)
    k2, c2, io2, key2, yb = kb.parts(xb, sb, out_mc, cores=cb_)
    prog = ttnn.ProgramDescriptor(kernels=k1 + k2, semaphores=[], cbs=c1 + c2)
    prog.custom_program_hash = _hash(("i2m_bconv_pair", key1, key2))
    ttnn.generic_op(list({id(t): t for t in io1 + io2}.values()), prog)
    return ya, yb
