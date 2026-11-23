# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Several decode projections with their own inputs in ONE program, each weight streamed per DRAM bank (a sliding or
global layer's q | r | s | g, k and v: three launches of ~11-16 us each on 4 dies -> one).

Each projection gets its own reader cores beside every bank (the qwen36 DecodeLinear placement, later projections
excluding the earlier ones' cores) and its own per-bank column-major weight copy (DecodeLinear.weight); the kernels are
the LM head's (kernels/head_compute.cpp without the softcap; kernels/mstream_writer.cpp: FP32 or BF16 output tiles).
One tile row (<= 32 rows) per call."""

from __future__ import annotations

import ttnn
from models.demos.blackhole.qwen36.tt.decode_linear.ops import KERNEL_DIR as DL_DIR
from models.demos.blackhole.qwen36.tt.decode_linear.ops import DecodeLinear

from .head import KDIR

OB = {ttnn.float32: 4096, ttnn.bfloat16: 2048}


class MultiStream:
    def __init__(self, mesh, k, specs, weight_dtype, interleaved=False):
        """specs: [(n output columns, output dtype, readers per bank)] in order. interleaved: stream the model's own
        interleaved [K, n] weights in place (n a multiple of 8 tiles: each tile column sits in one bank,
        kernels/ilstream_reader.cpp) instead of per-bank copies (weights())."""
        self.mesh, self.k, self.wdt, self.il = mesh, k, weight_dtype, interleaved
        self.parts, used = [], set()
        for n, dt, r in specs:
            assert not interleaved or n % 256 == 0, n
            dl = DecodeLinear(mesh, k, n, weight_dtype, readers=r, exclude=tuple(used))
            used |= {(c.x, c.y) for c, _, _, _ in dl.assign}
            self.parts.append((dl, dt))

    def weights(self, per_part):
        """per_part: for each projection, one [K, n] torch weight per device -> the decode copies."""
        return [dl.weight(pd) for (dl, _), pd in zip(self.parts, per_part)]

    def __call__(self, xs, ws, rows, out_mc=ttnn.DRAM_MEMORY_CONFIG):
        """xs: one [1, 1, rows, K] bf16 TILE input per projection (may repeat), ws = weights() -> outputs
        [1, 1, rows, n] in the specs' dtypes."""
        assert 1 <= rows <= 32
        cb_x, cb_w, cb_out = 0, 1, 16
        dm = ttnn.DataMovementConfigDescriptor
        kernels, cbs, outs, io = [], [], [], []
        for (dl, dt), x, w in zip(self.parts, xs, ws):
            out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, rows, dl.n]), dt, ttnn.TILE_LAYOUT, self.mesh, out_mc)
            outs.append(out)
            io += [x, w, out]
            fmt = lambda i, d, page, pages: ttnn.CBDescriptor(  # noqa: E731
                total_size=page * pages, core_ranges=dl.cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=d, page_size=page)])
            cbs += [fmt(cb_x, ttnn.bfloat16, 2048, dl.kt), fmt(cb_w, self.wdt, dl.tile_bytes, dl.buffers * dl.kt),
                    fmt(cb_out, dt, OB[dt], 2)]
            rd_ct = [cb_w, dl.kt, dl.tile_bytes, dl.n_tiles // 8 if self.il else dl.page_bytes, dl.buffers]
            wr_ct = [cb_x, cb_out, dl.kt, rows, dl.n_tiles, OB[dt], 8 if self.il else 1]
            wr_ct += ttnn.TensorAccessorArgs(x).get_compile_time_args() + ttnn.TensorAccessorArgs(out).get_compile_time_args()
            cp_rt = ttnn.RuntimeArgs()
            rd_args, wr_args = {}, {}
            for c, b, c0, nc in dl.assign:
                rd_args[(c.x, c.y)] = [w.buffer_address(), b, c0, nc, (b + c0) & 0x3]
                col0 = b + 8 * c0 if self.il else b * dl.C + c0
                wr_args[(c.x, c.y)] = [x.buffer_address(), out.buffer_address(), col0, nc]
                cp_rt[c.x][c.y] = [nc]

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
                kernel_source=str(KDIR / "head_compute.cpp"), core_ranges=dl.cores,
                compile_time_args=[cb_x, cb_w, cb_out, dl.kt, 0, 0], runtime_args=cp_rt,
                config=ttnn.ComputeConfigDescriptor(math_fidelity=dl.fidelity, fp32_dest_acc_en=True,
                                                    math_approx_mode=False)))
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return outs
