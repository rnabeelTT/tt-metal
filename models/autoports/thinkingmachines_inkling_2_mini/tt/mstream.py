# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Several decode projections with their own inputs in ONE program, each weight streamed per DRAM bank (a sliding or
global layer's q | r | s | g, k and v: three launches of ~11-16 us each on 4 dies -> one).

Each projection (or group of projections) gets its own reader cores beside every bank (the qwen36 DecodeLinear
placement, later groups excluding the earlier ones' cores) and streams either the model's interleaved weight in place
(kernels/ilstream_reader.cpp) or a per-bank column-major copy (DecodeLinear.weight); kernels/ms_compute.cpp (the LM
head's matmul without the softcap, in column groups) and kernels/mstream_writer.cpp (FP32 or BF16 output tiles).
One tile row (<= 32 rows) per call."""

from __future__ import annotations

import ttnn
from models.demos.blackhole.qwen36.tt.decode_linear.ops import KERNEL_DIR as DL_DIR
from models.demos.blackhole.qwen36.tt.decode_linear.ops import DecodeLinear

from .head import KDIR

OB = {ttnn.float32: 4096, ttnn.bfloat16: 2048}
TINY = __import__("os").environ.get("I2M_MS_TINY", "1") == "1"
TP4 = __import__("os").environ.get("I2M_MS_TP4", "1") == "1"
GRETUNE = __import__("os").environ.get("I2M_MS_GRETUNE", "1") == "1"


def knobs(ndev, kt):
    """(hmin, g, kb, nblk) of the interleaved streams; env I2M_MS_HMIN / G / KB / NBLK override. hmin: the shortest
    activation / output CB tile (4 rows: bitwise, a little less math than 8); g: columns per group (one matmul_block per
    k for the group: one activation unpack for g weight tiles); kb: k rows per pushed block; nblk: ring blocks (nblk - 1
    in flight). <= 2 dies: 4 / 4 / 8 / 7 (one die 64.5 -> 57.0 us per call with the k | v group, TP2 42.6 -> 37.3); 4
    dies (I2M_MS_TP4, default 1; 0: the previous kernels' 8 / 1 / kt / 3) the same. I2M_MS_GRETUNE (default 1;
    0: g = 4): g = 3 on one die, 1 on 2 and 4 dies (traced per call on one die at each mesh's per-die
    shapes, tools/bench_mstream.py: one die 55.3 -> 51.8 us, TP2 29.0 -> 26.7, TP4 16.2 -> 15.3; bitwise)."""
    env = lambda k, d: int(__import__("os").environ.get(f"I2M_MS_{k}", d))  # noqa: E731
    tuned = ndev <= 2 or TP4
    g = (3 if ndev == 1 else 1) if GRETUNE and tuned else 4 if tuned else 1
    return (env("HMIN", 4 if tuned else 8), env("G", g), env("KB", 8 if tuned else kt),
            env("NBLK", 7 if tuned else 3))


class MultiStream:
    def __init__(self, mesh, k, specs, weight_dtype, interleaved=False, fidelity=ttnn.MathFidelity.HiFi2, groups=None):
        """specs: [(n output columns, output dtype, readers per bank)] in order. interleaved: stream the model's own
        interleaved [K, n] weights in place (n a multiple of 8 tiles: each tile column sits in one bank,
        kernels/ilstream_reader.cpp) instead of per-bank copies (weights()). groups: lists of spec indices sharing one
        set of reader cores (equal n and dtype; interleaved only): each bank's columns of the members back to back,
        split over the first member's readers (k and v: one core per bank streams both, so the cores per bank hold
        about equal columns)."""
        self.mesh, self.k, self.wdt, self.il = mesh, k, weight_dtype, interleaved
        self.groups = groups or [[i] for i in range(len(specs))]
        assert sorted(i for g in self.groups for i in g) == list(range(len(specs)))
        self.parts, used = [], set()
        for g in self.groups:
            n, dt, r = specs[g[0]]
            assert all(specs[i][:2] == (n, dt) for i in g) and (interleaved or len(g) == 1)
            assert not interleaved or n % 256 == 0, n
            dl = DecodeLinear(mesh, k, n * len(g), weight_dtype, readers=r, fidelity=fidelity, exclude=tuple(used))
            used |= {(c.x, c.y) for c, _, _, _ in dl.assign}
            self.parts.append((dl, dt, g, n))

    def weights(self, per_part):
        """per_part: for each projection, one [K, n] torch weight per device -> the decode copies."""
        return [dl.weight(pd) for (dl, _, _, _), pd in zip(self.parts, per_part)]

    def __call__(self, xs, ws, rows, out_mc=ttnn.DRAM_MEMORY_CONFIG):
        """xs: one [1, 1, rows, K] bf16 TILE input per projection (may repeat), ws = weights() -> outputs
        [1, 1, rows, n] in the specs' dtypes."""
        assert 1 <= rows <= 32
        cb_x, cb_w, cb_out = 0, 1, 16
        hmin, G, KB, NBLK = knobs(self.mesh.get_num_devices(), self.k // 32)
        # activation / output CB tiles 4 / 8 / 16 rows tall when `rows` fits (in0 h x 32: 8 rows a partial face, 4 FPU
        # passes per tile instead of 16; 16 rows: 8): the math scales with the live rows, each row's sum unchanged
        h = next(v for v in (hmin, 8, 16, 32) if v >= rows) if TINY else 32
        dm = ttnn.DataMovementConfigDescriptor
        kernels, cbs, io = [], [], []
        outs = [None] * len(xs)
        for dl, dt, g, n in self.parts:
            gout = [ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, rows, n]), dt, ttnn.TILE_LAYOUT, self.mesh, out_mc)
                    for _ in g]
            for i, o in zip(g, gout):
                outs[i] = o
                io += [xs[i], ws[i], o]
            x0, nt = xs[g[0]], n // 32
            fmt = lambda i, d, page, pages, th=32: ttnn.CBDescriptor(  # noqa: E731
                total_size=page * pages, core_ranges=dl.cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=d, page_size=page,
                                                            tile=ttnn.TileDescriptor(th, 32) if th < 32 else None)])
            kb = KB if dl.kt % KB == 0 else dl.kt
            gw = G if self.il else 1
            cbs += [fmt(cb_x, ttnn.bfloat16, 64 * h, dl.kt * len(g), h),
                    fmt(cb_w, self.wdt, dl.tile_bytes, NBLK * kb * gw if self.il else dl.buffers * dl.kt),
                    fmt(cb_out, dt, OB[dt] * h // 32, 2 * gw, h)]
            rd_ct = ([cb_w, dl.kt, dl.tile_bytes, nt // 8, NBLK, kb, gw] if self.il else
                     [cb_w, dl.kt, dl.tile_bytes, dl.page_bytes, dl.buffers])
            wr_ct = [cb_x, cb_out, dl.kt, rows, nt, OB[dt], 8 if self.il else 1, h]
            wr_ct += ttnn.TensorAccessorArgs(x0).get_compile_time_args()
            wr_ct += ttnn.TensorAccessorArgs(gout[0]).get_compile_time_args()
            cp_rt = ttnn.RuntimeArgs()
            rd_args, wr_args = {}, {}
            cm = dl.C // len(g)  # each member's columns per bank
            for c, b, c0, nc in dl.assign:
                segs = []  # (member, column within the member's bank columns, count)
                for m in range(len(g)):
                    lo, hi = max(c0, m * cm), min(c0 + nc, (m + 1) * cm)
                    if hi > lo:
                        segs.append((m, lo - m * cm, hi - lo))
                if not self.il:  # (per-bank copies: one member, the old dl_reader arguments)
                    rd_args[(c.x, c.y)] = [ws[g[0]].buffer_address(), b, c0, nc, (b + c0) & 0x3]
                else:
                    rd_args[(c.x, c.y)] = [b, (b + c0) & 0x3, len(segs)]
                    for m, cc, k in segs:
                        rd_args[(c.x, c.y)] += [ws[g[m]].buffer_address(), cc, k]
                wr_args[(c.x, c.y)] = [len(segs)]
                for m, cc, k in segs:
                    col0 = b + 8 * cc if self.il else b * dl.C + cc
                    wr_args[(c.x, c.y)] += [xs[g[m]].buffer_address(), gout[m].buffer_address(), col0, k]
                cp_rt[c.x][c.y] = [len(segs)] + [k for _, _, k in segs]

            def rt_for(args, cores):
                rt = ttnn.RuntimeArgs()
                for c in ttnn.corerange_to_cores(cores, row_wise=True):
                    rt[c.x][c.y] = args[(c.x, c.y)]
                return rt

            for cores, rnoc, wnoc in ((dl.noc0_cores, ttnn.NOC.NOC_0, ttnn.NOC.NOC_1),
                                      (dl.noc1_cores, ttnn.NOC.NOC_1, ttnn.NOC.NOC_0)):
                if cores is None:
                    continue
                kernels += [
                    ttnn.KernelDescriptor(kernel_source=str(KDIR / "ilstream_reader.cpp") if self.il
                                          else str(DL_DIR / "dl_reader.cpp"), core_ranges=cores,
                                          compile_time_args=rd_ct, runtime_args=rt_for(rd_args, cores),
                                          config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=rnoc)),
                    ttnn.KernelDescriptor(kernel_source=str(KDIR / "mstream_writer.cpp"), core_ranges=cores,
                                          compile_time_args=wr_ct, runtime_args=rt_for(wr_args, cores),
                                          config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=wnoc)),
                ]
            kernels.append(ttnn.KernelDescriptor(
                kernel_source=str(KDIR / "ms_compute.cpp"), core_ranges=dl.cores,
                compile_time_args=[cb_x, cb_w, cb_out, dl.kt, kb if self.il else dl.kt, gw], runtime_args=cp_rt,
                config=ttnn.ComputeConfigDescriptor(math_fidelity=dl.fidelity, fp32_dest_acc_en=True,
                                                    math_approx_mode=False)))
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return outs
