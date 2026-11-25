# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Many page-range copies between interleaved DRAM tensors in one program (tt/kernels/blkcopy.cpp).

A job (src, src_page, dst, dst_page, n_pages) copies pages [src_page, + n) of src to [dst_page, + n) of dst (same page
size; for a TILE tensor a page is a tile, for a ROW_MAJOR one a row). Used by the prefix cache (tt/prefix.py) to move
the sliding-window KV blocks and conv states of a sequence into / out of the snapshot pools: hundreds of small copies
in one launch instead of hundreds of ttnn ops. The runtime-argument layout is fixed (MAXJ job slots per core), so one
program serves every call: it is compiled at warm-up, before any trace capture."""

from __future__ import annotations

import hashlib

import ttnn

from .bconv import KDIR, _cb, _range_set, _rt
from .bconv import _ks

NCORES = 64
MAXJ = 40  # job slots per core (6 runtime args each)
SPLIT = 64  # pages per job at most (balance over the cores)
BUF = 32768  # L1 staging bytes per core


def page_bytes(t) -> int:
    """Bytes per page of a DRAM interleaved tensor (tile or row-major row) on one device."""
    if t.layout == ttnn.TILE_LAYOUT:
        return {ttnn.bfloat16: 2048, ttnn.float32: 4096, ttnn.bfloat8_b: 1088, ttnn.int32: 4096}[t.dtype]
    es = {ttnn.bfloat16: 2, ttnn.float32: 4, ttnn.int32: 4, ttnn.uint32: 4}[t.dtype]
    return int(t.padded_shape[-1]) * es


class BlockCopy:
    def __init__(self, mesh, proto):
        """proto: any DRAM interleaved tensor (the accessor arguments are the same for all of them)."""
        self.mesh = mesh
        grid = mesh.compute_with_storage_grid_size()
        self.cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(min(NCORES, grid.x * grid.y))]
        self.crs = _range_set(self.cores)
        self.ct = ttnn.TensorAccessorArgs(proto).get_compile_time_args()
        self.proto = proto

    def __call__(self, jobs):
        """jobs: [(src, src_page, dst, dst_page, n_pages)] (tensors on this mesh, same page size per job)."""
        work = []
        for src, sp, dst, dp, n in jobs:
            for t in (src, dst):
                mc = t.memory_config()
                assert mc.buffer_type == ttnn.BufferType.DRAM and not t.is_sharded(), f"blkcopy: {mc} (DRAM only)"
            ps = page_bytes(src)
            assert ps == page_bytes(dst) and ps <= BUF, (ps, page_bytes(dst))
            for o in range(0, int(n), SPLIT):
                work.append((src.buffer_address(), int(sp) + o, dst.buffer_address(), int(dp) + o, min(SPLIT, int(n) - o), ps))
        if not work:
            return
        nc = len(self.cores)
        assert len(work) <= nc * MAXJ, len(work)
        per = [[] for _ in range(nc)]
        for i, w in enumerate(work):
            per[i % nc].append(w)
        rt = []
        for lst in per:
            a = [len(lst)]
            for w in lst:
                a += list(w)
            rt.append(a + [0] * (1 + 6 * MAXJ - len(a)))
        k = ttnn.KernelDescriptor(
            kernel_source=_ks(KDIR, "blkcopy.cpp"), core_ranges=self.crs, compile_time_args=[0, BUF] + self.ct,
            runtime_args=_rt(self.cores, rt),
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0))
        prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[_cb(self.crs, 0, ttnn.bfloat16, 2048, BUF // 2048)])
        key = ("i2m_blkcopy", nc, MAXJ, BUF)
        prog.custom_program_hash = int.from_bytes(hashlib.sha1(repr(key).encode()).digest()[:8], "little") & ((1 << 63) - 1)
        ttnn.generic_op([self.proto, self.proto], prog)
