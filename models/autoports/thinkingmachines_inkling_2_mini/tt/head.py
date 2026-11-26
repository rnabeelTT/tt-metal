# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The decode LM head streamed per DRAM bank, softcap included: logits = c tanh(h W / c), FP32, in one program.

The weight placement and reader are the qwen36 DecodeLinear's (a per-bank column-major copy of the weight, `readers`
cores per bank streaming their columns in 16 KB packets); the compute (matmul + the softcap on the SFPU) and the
writer (FP32 logits pages) are this port's (kernels/head_*.cpp). One tile row (<= 32 rows) per call."""

from __future__ import annotations

import os
import struct
from pathlib import Path

import ttnn
from models.demos.blackhole.qwen36.tt.decode_linear.ops import KERNEL_DIR as DL_DIR
from models.demos.blackhole.qwen36.tt.decode_linear.ops import DecodeLinear
from .bconv import _ks

KDIR = Path(__file__).parent / "kernels"
# I2M_HEAD_TOPK (default 1): the writers also keep each live row's candidate list per core as the tiles stream
# (kernels/lhlist.hpp; the sampling tail's top-k then merges these lists, tt/sampling.py HeadTopK, instead of scanning
# the logits again), for calls of up to I2M_HEAD_TOPK_W rows; default by dies (the writer's list work grows with the
# rows and its columns: one die traced B1 15.53 -> 15.42 ms, B4 30.02 -> 29.87, verify 29.38 -> 29.28, B8 +0.05; TP2 B1
# 11.07 -> 11.02, B4 flat, verify +0.03; TP4 B1 8.845 -> 8.815, B4 +0.09, verify +0.07)
HEAD_TOPK = os.environ.get("I2M_HEAD_TOPK", "1") == "1"
HEAD_TOPK_W = int(os.environ.get("I2M_HEAD_TOPK_W", "0"))
TOPK_W_DEFAULT = {1: 4, 2: 1, 4: 1}
LIST_PAGE, LIST_CAP = 576, 192  # (lhlist.hpp LH_PAGE, LH_CAP)
# I2M_HEAD_TOPK_HEAP (default 1): the list as a heap (else buffered + compacted); I2M_HEAD_OUT_PAGES (default 8): logits
# tiles the writer may lag behind the compute (its list work comes in bursts at the start)
TOPK_HEAP = os.environ.get("I2M_HEAD_TOPK_HEAP", "1")
OUT_PAGES = int(os.environ.get("I2M_HEAD_OUT_PAGES", "8"))
# I2M_HEAD_LIVE (default 1): the logits pages get their live rows only (every consumer reads rows < rows: the top-k
# programs, the host readback's untilize_with_unpadding; the other rows were don't-care values already)
LIVE = os.environ.get("I2M_HEAD_LIVE", "1") == "1"
# I2M_HEAD_STAGE: with LIVE, the live rows staged in L1 and written this many tiles at a time (0: a write per tile);
# default 64 tiles up to 8 rows, 16 up to 16 (<= 64 KB of L1), none above (one die head + tail, per tile -> staged:
# W1 1107 -> 1071 us, W2 1183 -> 1143, W4 1352 -> 1334, W8 1903 -> 1851, W16 3231 -> 3224)
STAGE = int(os.environ.get("I2M_HEAD_STAGE", "-1"))
# the lists of the last head call per logits buffer address: {address: (head, rows)} (the tail looks its logits up)
LISTS = {}


def _bits(f):
    return struct.unpack("<I", struct.pack("<f", f))[0]


class StreamHead(DecodeLinear):
    def __init__(self, mesh_device, k, n, weight_dtype, softcap, readers=2, fidelity=ttnn.MathFidelity.HiFi2,
                 buffers=2):
        super().__init__(mesh_device, k, n, weight_dtype, readers=readers, fidelity=fidelity, buffers=buffers)
        # (2 buffers with one packet per column: the reader waits on the read it just issued and pushes columns before
        # they land; measured: K <= 480 bfp8 logits wrong and run-dependent, K >= 768 and 3+ buffers bitwise stable)
        assert buffers >= 3 or self.kt * self.tile_bytes > self.page_bytes, (k, buffers)
        self.softcap = softcap
        # I2M_HEAD_SC (default 1): the softcap on the live rows only (rows <= 16: faces 0 / 1; an SFPU row group is 4
        # rows x 8 columns, two per 4 rows) instead of the whole tile; the same SFPU functions, so the live logits are
        # bitwise unchanged
        self.sc_rows = os.environ.get("I2M_HEAD_SC", "1") == "1"
        self.lists = None  # [cores x 32 rows] list pages, DRAM, persistent
        self.slots = [(b * self.C + c0, nc) for _, b, c0, nc in self.assign]  # per core: first tile, tiles

    def alloc_lists(self):
        """the persistent list buffer (before any trace capture: the first eager call)"""
        if self.lists is None:
            self.lists = ttnn.allocate_tensor_on_device(ttnn.Shape([len(self.assign) * 32, LIST_PAGE // 4]), ttnn.uint32,
                                                        ttnn.ROW_MAJOR_LAYOUT, self.mesh_device, ttnn.DRAM_MEMORY_CONFIG)
        return self.lists

    def __call__(self, x, w, rows, out_memory_config=ttnn.DRAM_MEMORY_CONFIG, out=None):
        """x [1, 1, rows, K] bf16 TILE (rows <= 32), w = weight([W]) -> softcapped logits [1, 1, rows, N] fp32 TILE
        (out: an existing tensor of that spec to write them into)."""
        assert 1 <= rows <= 32
        if out is None:
            out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, rows, self.n]), ttnn.float32, ttnn.TILE_LAYOUT,
                                                 self.mesh_device, out_memory_config)
        topk = HEAD_TOPK and rows <= (HEAD_TOPK_W or TOPK_W_DEFAULT.get(self.mesh_device.get_num_devices(), 4))
        cb_x, cb_w, cb_out, cb_l = 0, 1, 16, 2
        fmt = lambda i, dt, page, pages: ttnn.CBDescriptor(  # noqa: E731
            total_size=page * pages, core_ranges=self.cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)])
        cbs = [fmt(cb_x, ttnn.bfloat16, 2048, self.kt), fmt(cb_w, self.weight_dtype, self.tile_bytes, self.buffers * self.kt),
               fmt(cb_out, ttnn.float32, 4096, OUT_PAGES if topk else 2)]
        stage = 0 if not LIVE or rows > 16 else (STAGE if STAGE >= 0 else 64 if rows <= 8 else 16)
        if stage:
            cbs.append(fmt(3, ttnn.float32, stage * 2 * rows * 64, 1))
        if topk:
            lists = self.alloc_lists()
            cbs.append(fmt(cb_l, ttnn.float32, rows * ((2 * LIST_CAP + 16) * 4 + LIST_PAGE) + LIST_CAP * 4, 1))
        rd_ct = [cb_w, self.kt, self.tile_bytes, self.page_bytes, self.buffers]
        wr_ct = [cb_x, cb_out, self.kt, rows, self.n_tiles]
        wr_ct += ttnn.TensorAccessorArgs(x).get_compile_time_args() + ttnn.TensorAccessorArgs(out).get_compile_time_args()
        if topk:
            wr_ct += [cb_l] + ttnn.TensorAccessorArgs(lists).get_compile_time_args()
        cp_ct = [cb_x, cb_w, cb_out, self.kt, _bits(1.0 / self.softcap), _bits(self.softcap)]
        cp_rt = ttnn.RuntimeArgs()
        rd_args, wr_args = {}, {}
        for i, (c, b, c0, nc) in enumerate(self.assign):
            rd_args[(c.x, c.y)] = [w.buffer_address(), b, c0, nc, (b + c0) & 0x3]
            wr_args[(c.x, c.y)] = [x.buffer_address(), out.buffer_address(), b * self.C + c0, nc] + (
                [lists.buffer_address(), i] if topk else [])
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
                ttnn.KernelDescriptor(kernel_source=_ks(DL_DIR, "dl_reader.cpp"), core_ranges=cores,
                                      compile_time_args=rd_ct, runtime_args=rt_for(rd_args, cores),
                                      config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=rnoc)),
                ttnn.KernelDescriptor(kernel_source=_ks(KDIR, "head_writer.cpp"), core_ranges=cores,
                                      compile_time_args=wr_ct, runtime_args=rt_for(wr_args, cores),
                                      defines=([("HEAD_TOPK", "1"), ("LH_HEAP", TOPK_HEAP)] if topk else []) + [("HEAD_LIVE", str(int(LIVE))), ("HEAD_STAGE", str(stage))],
                                      config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=wnoc)),
            ]
        kernels.append(ttnn.KernelDescriptor(
            kernel_source=_ks(KDIR, "head_compute.cpp"), core_ranges=self.cores, compile_time_args=cp_ct,
            runtime_args=cp_rt,
            defines=[("HEAD_SOFTCAP", "1")] + ([("HEAD_SC_ITERS", str(2 * ((rows + 3) // 4)))]
                                               if self.sc_rows and rows <= 16 else []),
            config=ttnn.ComputeConfigDescriptor(math_fidelity=self.fidelity, fp32_dest_acc_en=True,
                                                math_approx_mode=False)))
        ttnn.generic_op([x, w, out] + ([lists] if topk else []),
                        ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        if topk:
            LISTS[out.buffer_address()] = (self, rows)
        else:
            LISTS.pop(out.buffer_address(), None)
        return out
