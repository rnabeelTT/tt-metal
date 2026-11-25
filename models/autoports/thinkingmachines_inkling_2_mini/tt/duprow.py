# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Row 0 -> row 1 of every tile of a one-row bf16 [1, 1, 1, H] TILE tensor, in place (tt/kernels/duprow.cpp): the
fused MoE runs one decode row through its 2-row path with the row twice (its single-row router is wrong for the sigb
router); this replaces the concat [x, x] (11 us) with a ~2 us program."""

from __future__ import annotations

import hashlib

import ttnn

from .bconv import KDIR, _cb, _range_set, _rt
from .bconv import _ks

CORES = 8


class DupRow:
    def __init__(self, mesh, H: int):
        self.mesh, self.Ht = mesh, H // 32
        grid = mesh.compute_with_storage_grid_size()
        nc = CORES
        while self.Ht % nc:
            nc -= 1
        self.cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(nc)]
        self.per = self.Ht // nc
        self.crs = _range_set(self.cores)

    def __call__(self, x):
        assert x.dtype == ttnn.bfloat16 and x.layout == ttnn.TILE_LAYOUT and not x.is_sharded()
        nc = len(self.cores)
        k = ttnn.KernelDescriptor(
            kernel_source=_ks(KDIR, "duprow.cpp"), core_ranges=self.crs,
            compile_time_args=[0] + ttnn.TensorAccessorArgs(x).get_compile_time_args(),
            runtime_args=_rt(self.cores, [[x.buffer_address(), i * self.per, self.per] for i in range(nc)]),
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0))
        prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[_cb(self.crs, 0, ttnn.bfloat16, 2048, self.per)])
        key = ("i2m_duprow", self.Ht, nc, repr(x.memory_config()))
        prog.custom_program_hash = int.from_bytes(hashlib.sha1(repr(key).encode()).digest()[:8], "little") & ((1 << 63) - 1)
        ttnn.generic_op([x, x], prog)
        return x
