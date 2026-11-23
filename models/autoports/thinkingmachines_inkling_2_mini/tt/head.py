# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The decode LM head streamed per DRAM bank, softcap included: logits = c tanh(h W / c), FP32, in one program.

The weight placement and reader are the qwen36 DecodeLinear's (a per-bank column-major copy of the weight, `readers`
cores per bank streaming their columns in 16 KB packets); the compute (matmul + the softcap on the SFPU) and the
writer (FP32 logits pages) are this port's (kernels/head_*.cpp). One tile row (<= 32 rows) per call."""

from __future__ import annotations

import struct
from pathlib import Path

import ttnn
from models.demos.blackhole.qwen36.tt.decode_linear.ops import KERNEL_DIR as DL_DIR
from models.demos.blackhole.qwen36.tt.decode_linear.ops import DecodeLinear

KDIR = Path(__file__).parent / "kernels"


def _bits(f):
    return struct.unpack("<I", struct.pack("<f", f))[0]


class StreamHead(DecodeLinear):
    def __init__(self, mesh_device, k, n, weight_dtype, softcap, readers=2):
        super().__init__(mesh_device, k, n, weight_dtype, readers=readers)
        self.softcap = softcap

    def __call__(self, x, w, rows, out_memory_config=ttnn.DRAM_MEMORY_CONFIG):
        """x [1, 1, rows, K] bf16 TILE (rows <= 32), w = weight([W]) -> softcapped logits [1, 1, rows, N] fp32 TILE."""
        assert 1 <= rows <= 32
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, rows, self.n]), ttnn.float32, ttnn.TILE_LAYOUT,
                                             self.mesh_device, out_memory_config)
        cb_x, cb_w, cb_out = 0, 1, 16
        fmt = lambda i, dt, page, pages: ttnn.CBDescriptor(  # noqa: E731
            total_size=page * pages, core_ranges=self.cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)])
        cbs = [fmt(cb_x, ttnn.bfloat16, 2048, self.kt), fmt(cb_w, self.weight_dtype, self.tile_bytes, self.buffers * self.kt),
               fmt(cb_out, ttnn.float32, 4096, 2)]
        rd_ct = [cb_w, self.kt, self.tile_bytes, self.page_bytes, self.buffers]
        wr_ct = [cb_x, cb_out, self.kt, rows, self.n_tiles]
        wr_ct += ttnn.TensorAccessorArgs(x).get_compile_time_args() + ttnn.TensorAccessorArgs(out).get_compile_time_args()
        cp_ct = [cb_x, cb_w, cb_out, self.kt, _bits(1.0 / self.softcap), _bits(self.softcap)]
        cp_rt = ttnn.RuntimeArgs()
        rd_args, wr_args = {}, {}
        for c, b, c0, nc in self.assign:
            rd_args[(c.x, c.y)] = [w.buffer_address(), b, c0, nc, (b + c0) & 0x3]
            wr_args[(c.x, c.y)] = [x.buffer_address(), out.buffer_address(), b * self.C + c0, nc]
            cp_rt[c.x][c.y] = [nc]

        def rt_for(args, cores):
            rt = ttnn.RuntimeArgs()
            for c in ttnn.corerange_to_cores(cores, row_wise=True):
                rt[c.x][c.y] = args[(c.x, c.y)]
            return rt

        dm = ttnn.DataMovementConfigDescriptor
        kernels = []
        for cores, rnoc, wnoc in ((self.noc0_cores, ttnn.NOC.NOC_0, ttnn.NOC.NOC_1),
                                  (self.noc1_cores, ttnn.NOC.NOC_1, ttnn.NOC.NOC_0)):
            if cores is None:
                continue
            kernels += [
                ttnn.KernelDescriptor(kernel_source=str(DL_DIR / "dl_reader.cpp"), core_ranges=cores,
                                      compile_time_args=rd_ct, runtime_args=rt_for(rd_args, cores),
                                      config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=rnoc)),
                ttnn.KernelDescriptor(kernel_source=str(KDIR / "head_writer.cpp"), core_ranges=cores,
                                      compile_time_args=wr_ct, runtime_args=rt_for(wr_args, cores),
                                      config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=wnoc)),
            ]
        kernels.append(ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "head_compute.cpp"), core_ranges=self.cores, compile_time_args=cp_ct,
            runtime_args=cp_rt, defines=[("HEAD_SOFTCAP", "1")],
            config=ttnn.ComputeConfigDescriptor(math_fidelity=self.fidelity, fp32_dest_acc_en=True,
                                                math_approx_mode=False)))
        ttnn.generic_op([x, w, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return out
