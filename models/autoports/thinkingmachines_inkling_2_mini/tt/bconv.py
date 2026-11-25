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
# I2M_BCONV_LIVE (with ROWS): x / states / taps read and y / states written on the live rows only (faces 0 / 1), not
# whole fp32 tiles: the conv pair moved ~2.8 MB per application, DRAM-bound
LIVE = __import__("os").environ.get("I2M_BCONV_LIVE", "1") == "1"
# I2M_BCONV_ONE: bconv_pair's two convs as one kernel group (their kernels are identical): the cores started ~2 us after
# their firmware with two groups
ONE_GROUP = __import__("os").environ.get("I2M_BCONV_ONE", "1") == "1"
TILE = 32
# speculative verify rows (tt/kernels/seqshift.hpp): (R, G, acc) while a verify pass runs / is captured, else None.
# acc: uint32 [1, 32] row-major, group g's accepted row of the previous pass.
SEQ = None
# the post / pre readers' verify rows for all their tiles with one barrier (seqshift.hpp seq_pre / seq_fin). I2M_SEQ_ONE
SEQ_ONE = __import__("os").environ.get("I2M_SEQ_ONE", "1") == "1"


def seq_parts(ct, nargs, cb, crs):
    """With SEQ set: the verify defines (ct: the reader's compile args, the acc accessor appended in place; nargs: the
    acc address's index in the reader's args; cb: a free CB index for the scratch tile) -> (defines, cbs, acc)."""
    if SEQ is None:
        return [], [], None
    R, G, acc = SEQ
    defs = [("SEQ_R", str(R)), ("SEQ_G", str(G)), ("SEQ_CT", str(len(ct))), ("SEQ_ARG", str(nargs)),
            ("SEQ_CB", str(cb)), ("SEQ_ONE", str(int(SEQ_ONE)))]
    ct += ttnn.TensorAccessorArgs(acc).get_compile_time_args()
    return defs, [_cb(crs, cb, ttnn.float32, 4096, 1)], acc


_RS = {}
_KS = {}


def _ks(d, name):
    """str(d / name), memoized (the pathlib join ran on every program build: ~8 us each, ~1600 per prefill chunk)"""
    key = (d, name)
    r = _KS.get(key)
    if r is None:
        r = _KS[key] = str(d / name)
    return r


def _range_set(cores):
    """the cores' CoreRangeSet (memoized: the merge loop costs ~75 us at 108 cores, run on every program build)"""
    key = tuple((c.x, c.y) for c in cores)
    rs = _RS.get(key)
    if rs is None:
        rs = ttnn.CoreRangeSet([ttnn.CoreRange(cores[0], cores[0])])
        for c in cores[1:]:
            rs = rs.merge(ttnn.CoreRangeSet([ttnn.CoreRange(c, c)]))
        _RS[key] = rs
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

    def step(self, x, s, out_mc=ttnn.L1_MEMORY_CONFIG, taps=None, dyn=None):
        kernels, cbs, io, key, y = self.parts(x, s, out_mc, taps, dyn=dyn)
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        prog.custom_program_hash = _hash(key)
        ttnn.generic_op(io, prog)
        return y

    def parts(self, x, s, out_mc=ttnn.L1_MEMORY_CONFIG, taps=None, cores=None, dyn=None):
        """cores: this conv's cores (default its own). x [1, 1, B, C] TILE interleaved (bf16 or fp32), s = [s0, s1, s2] fp32 TILE (updated in place) -> y bf16.
        taps: (w0, w1, w2, wc) [1, 1, B, C] fp32 per-row taps (the dynamic convs) instead of the static ones, or
        (cat,): one [1, 1, B, 4 C] tensor [w0 | w1 | w2 | wc].
        dyn = (d, e, sc): the dynamic convs' taps made here: tap j = d @ e_j + sc_j (d [1, 1, B, 32] bf16 per-head
        lags, e [1, 1, 32, 4 C] bf16 0 / 1 expansion, sc [1, 1, 1, 4 C] fp32 static taps; tap j at tile t + j C / 32),
        as the separate matmul + add did."""
        cores = self.cores if cores is None else cores
        crs = _range_set(cores)
        stride = 0
        if dyn is not None:
            taps = (dyn[1], dyn[2], dyn[0], dyn[0])  # (reader slots: e, sc, d; the last unused)
            stride = int(x.shape[-1]) // 32
        elif taps is not None and len(taps) == 1:
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
        ddefs = []
        if dyn is not None:  # (e's accessor above; sc's and d's follow)
            cb_d, cb_e, cb_sc = 8, 9, 10
            ddefs = [("BCONV_DYN", "1"), ("BCONV_B", str(int(x.shape[-2]))), ("BCONV_CT_SC", str(len(rd_ct))),
                     ("BCONV_CT_D", str(len(rd_ct) + len(ttnn.TensorAccessorArgs(dyn[2]).get_compile_time_args())))]
            rd_ct += ttnn.TensorAccessorArgs(dyn[2]).get_compile_time_args() \
                + ttnn.TensorAccessorArgs(dyn[0]).get_compile_time_args()
            cbs += [_cb(crs, cb_d, bf, 2048, 1), _cb(crs, cb_e, bf, 2048, 4), _cb(crs, cb_sc, f32, 4096, 4)]
        wr_ct = [cb_y, cb_o0, cb_o1, cb_o2] + ttnn.TensorAccessorArgs(y).get_compile_time_args() \
            + ttnn.TensorAccessorArgs(s[0]).get_compile_time_args()
        addr = [x.buffer_address()] + [t.buffer_address() for t in s] + [t.buffer_address() for t in w_taps] \
            + [wc.buffer_address()]
        sdefs, scbs, acc = seq_parts(rd_ct, len(addr) + 2, 7, crs)
        cbs += scbs
        rd_rt = [addr + [i * self.per, self.per] + ([acc.buffer_address()] if acc else []) for i in range(len(cores))]
        wr_rt = [[y.buffer_address()] + [t.buffer_address() for t in s] + [i * self.per, self.per]
                 for i in range(len(cores))]
        u2d = [ttnn.UnpackToDestMode.Default] * 64  # (this build: one entry per CB index, 64)
        for i in (cb_s, cb_w) + ((cb_x,) if x.dtype == ttnn.float32 else ()) + ((10,) if dyn is not None else ()):
            u2d[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                           dst_full_sync_en=True, math_approx_mode=False)
        cfg.unpack_to_dest_mode = u2d
        live = ROWS and LIVE and int(x.shape[-2]) <= 16
        ldefs = [("BCONV_LIVE", str(int(x.shape[-2])))] if live else []
        dmdefs = ([("BCONV_TAP_STRIDE", str(stride))] if stride else []) + sdefs + ddefs + ldefs
        cpdefs = ([("BCONV_ROWS", str(int(x.shape[-2])))] if ROWS and int(x.shape[-2]) <= 16 else []) + ddefs
        # (the pieces, so that bconv_pair can put two identical convs in one kernel group)
        self.spec = {"cores": list(cores), "cfg": cfg, "k": [
            ("bconv_reader.cpp", rd_ct, rd_rt, dmdefs, (ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0)),
            ("bconv_writer.cpp", wr_ct, wr_rt, dmdefs, (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1)),
            ("bconv_compute.cpp", [cb_x, cb_s, cb_w, cb_y, cb_o0, cb_o1, cb_o2], [[self.per]] * len(cores), cpdefs, None)],
            "sig": (str(x.dtype), dyn is None, tuple(c.total_size for c in cbs))}
        kernels = _kernels(self.spec)
        key = ("i2m_bconv", self.C, len(cores), cores[0].x, cores[0].y, str(x.dtype), int(x.shape[-2]), stride,
               repr(x.memory_config()), repr(out_mc), repr(s[0].memory_config()), repr(w_taps[0].memory_config()),
               SEQ[:2] + (SEQ_ONE,) if SEQ else None, live, None if dyn is None else (repr(dyn[0].memory_config()), repr(dyn[1].memory_config()),
                                                                    repr(dyn[2].memory_config())))
        io = list({id(t): t for t in [x] + list(s) + list(w_taps) + [wc, y] + ([acc] if acc else [])}.values())
        return kernels, cbs, io, key, y


def _kernels(spec):
    cores = spec["cores"]
    crs = _range_set(cores)
    out = []
    for src, ct, rt, defs, dmc in spec["k"]:
        cfg = spec["cfg"] if dmc is None else ttnn.DataMovementConfigDescriptor(processor=dmc[0], noc=dmc[1])
        out.append(ttnn.KernelDescriptor(kernel_source=_ks(KDIR, src), core_ranges=crs, compile_time_args=ct,
                                         runtime_args=_rt(cores, rt), defines=defs, config=cfg))
    return out


def _hash(key):
    return int.from_bytes(hashlib.sha1(repr(key).encode()).digest()[:8], "little") & ((1 << 63) - 1)


def bconv_pair(ka: "BConvKernel", xa, sa, kb: "BConvKernel", xb, sb, out_mc=ttnn.L1_MEMORY_CONFIG, dyn=(None, None)):
    """Two convs (a layer's k and v) in ONE program: b on the cores after a's -> (ya, yb). dyn: each one's
    parts(dyn=) (the global layers' dynamic taps)."""
    grid = ka.mesh.compute_with_storage_grid_size()
    n = len(ka.cores)
    assert n + len(kb.cores) <= grid.x * grid.y
    cb_ = [ttnn.CoreCoord((n + i) % grid.x, (n + i) // grid.x) for i in range(len(kb.cores))]
    k1, c1, io1, key1, ya = ka.parts(xa, sa, out_mc, dyn=dyn[0])
    sa_ = ka.spec
    k2, c2, io2, key2, yb = kb.parts(xb, sb, out_mc, cores=cb_, dyn=dyn[1])
    sb_ = kb.spec
    one = ONE_GROUP and all((x[0], list(x[1]), list(x[3])) == (y[0], list(y[1]), list(y[3]))
                            for x, y in zip(sa_["k"], sb_["k"])) and sa_["sig"] == sb_["sig"]
    if one:  # (identical kernels: one kernel group over both convs' cores, per-core arguments)
        merged = {"cores": sa_["cores"] + sb_["cores"], "cfg": sa_["cfg"],
                  "k": [(x[0], x[1], x[2] + y[2], x[3], x[4]) for x, y in zip(sa_["k"], sb_["k"])]}
        allc = _range_set(merged["cores"])
        cbs = [ttnn.CBDescriptor(total_size=c.total_size, core_ranges=allc, format_descriptors=c.format_descriptors)
               for c in c1]
        prog = ttnn.ProgramDescriptor(kernels=_kernels(merged), semaphores=[], cbs=cbs)
    else:
        prog = ttnn.ProgramDescriptor(kernels=k1 + k2, semaphores=[], cbs=c1 + c2)
    prog.custom_program_hash = _hash(("i2m_bconv_pair", key1, key2, one))
    ttnn.generic_op(list({id(t): t for t in io1 + io2}.values()), prog)
    return ya, yb
