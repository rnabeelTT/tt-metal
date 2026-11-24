# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode selection expansion (single path, HY4_IDX_PADSENT rows): one data-movement program instead of the ~12 ttnn
ops after the top-512 (to_layout, typecast, slice, multiply, repeat_interleave, add offsets, add base, typecast,
to_layout, slice + concat sentinels, concat tail, copy to the persistent selection):

    row r < R:  sel[r, 4 i + j] = 4 ids[r, i] + j + base   (i < 511, j < 4)       ; idx[r] = sel[r] ++ tail[r]
    row r >= R: sel[r] = 0xFFFFFFFF                                                 ; idx[r] = sel[r] ++ tail[r]

ids [1, 1, 32, 512] uint32 RM (topk_large_indices), base [1, 1, *, 1] fp32 TILE (an integer-valued float), tail
[1, 1, 32, tw] uint32 RM -> sel [1, 1, 32, 2044] (written into `sel` when given), idx [1, 1, 32, 2044 + tw] DRAM.
Integer-exact (the same values as the ttnn chain for ids < 2^22). One core per row, the two RISCs split the columns.
kernels/sel_expand.cpp."""

from __future__ import annotations

from pathlib import Path

import ttnn

KDIR = Path(__file__).parent / "kernels"


class SelExpandOp:
    _OPS = {}

    @classmethod
    def get(cls, mesh):
        if id(mesh) not in cls._OPS:
            cls._OPS[id(mesh)] = cls(mesh)
        return cls._OPS[id(mesh)]

    def __init__(self, mesh):
        self.mesh = mesh
        g = mesh.compute_with_storage_grid_size()
        self.grid = [ttnn.CoreCoord(x, y) for y in range(g.y) for x in range(g.x)]

    def __call__(self, ids, base, tail, R, sel=None, r=4, n_sel=511):
        rows = int(ids.shape[-2])
        K = int(ids.shape[-1])
        tw = int(tail.shape[-1])
        ns = n_sel * r
        assert rows == 32 and int(tail.shape[-2]) == rows and K >= n_sel
        if sel is None:
            sel = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, rows, ns]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT,
                                                 self.mesh, ttnn.DRAM_MEMORY_CONFIG)
        idx = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, rows, ns + tw]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT,
                                             self.mesh, ttnn.DRAM_MEMORY_CONFIG)
        cores = self.grid[:rows]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        half = (n_sel + 1) // 2  # groups per RISC
        # per RISC: ids row (K words) + the row being built (its half of sel) + the tail row
        cbsz = -(-(K * 4 + (half * r + tw) * 4 + 128 + max(tw * 4, 64) + 64) // 64) * 64
        cbs = [ttnn.CBDescriptor(total_size=cbsz, core_ranges=crs, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.uint32, page_size=cbsz)]) for i in (0, 1)]
        acc = lambda t: ttnn.TensorAccessorArgs(t).get_compile_time_args()  # noqa: E731
        ct = [K, n_sel, r, tw, int(R)] + acc(ids) + acc(base) + acc(tail) + acc(sel) + acc(idx)
        dm = ttnn.DataMovementConfigDescriptor
        kernels = []
        for part, (proc, noc) in enumerate(((ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
                                            (ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1))):
            g0, g1 = (0, half) if part == 0 else (half, n_sel)
            a = ttnn.RuntimeArgs()
            for row, c in enumerate(cores):
                a[c.x][c.y] = [ids.buffer_address(), base.buffer_address(), tail.buffer_address(), sel.buffer_address(),
                               idx.buffer_address(), row, g0, g1, part]
            kernels.append(ttnn.KernelDescriptor(kernel_source=str(KDIR / "sel_expand.cpp"), core_ranges=crs,
                                                 compile_time_args=ct, runtime_args=a, config=dm(processor=proc, noc=noc)))
        ttnn.generic_op([ids, base, tail, sel, idx], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return sel, idx
