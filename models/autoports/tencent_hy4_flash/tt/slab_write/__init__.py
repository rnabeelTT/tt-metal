# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Multi-slab row write: slabs of S rows of a row-major source -> a row-major destination at per-slab row offsets held
in a device uint32 tensor (written by the host before each trace replay). One program per call for any number of
slabs (one core each): the batched decode's KV / group-key slab writes of every user in one op."""

from __future__ import annotations

from pathlib import Path

import ttnn

KDIR = Path(__file__).parent / "kernels"


def _cb(cores, index, nbytes):
    fmt = ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.uint32, page_size=nbytes)
    return ttnn.CBDescriptor(total_size=nbytes, core_ranges=cores, format_descriptors=[fmt])


def slab_write(src, dst, offs, n_slabs: int, slab_rows: int = 32):
    """dst rows [offs[k], offs[k] + slab_rows) = src rows [k slab_rows, (k + 1) slab_rows), k < n_slabs.
    Row-major src / dst (same row width / dtype): any offsets. Tiled src / dst (slab_rows = 32): whole tile rows,
    offsets multiples of 32. offs [1, 1, 1, >= n_slabs] uint32 RM on the device."""
    tiled = src.layout == ttnn.TILE_LAYOUT
    assert dst.layout == src.layout
    if tiled:
        assert slab_rows == 32
        wt = int(src.shape[-1]) // 32
        pageb = 32 * 32 * src.element_size()
        S = wt
    else:
        pageb = int(src.shape[-1]) * src.element_size()
        assert pageb == int(dst.shape[-1]) * dst.element_size()
        S = slab_rows
    offb = -(-(int(offs.shape[-1]) * 4) // 32) * 32
    grid = src.device().compute_with_storage_grid_size()
    nc = min(n_slabs, grid.x * grid.y, int(__import__("os").environ.get("HY4_SLAB_CORES", "64")))
    cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(nc)]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    acc = lambda t: ttnn.TensorAccessorArgs(t).get_compile_time_args()  # noqa: E731
    ct = [S, pageb, offb, 0, 1, n_slabs, nc, int(tiled)] + acc(src) + acc(dst) + acc(offs)
    rt = ttnn.RuntimeArgs()
    for k, c in enumerate(cores):
        rt[c.x][c.y] = [src.buffer_address(), dst.buffer_address(), offs.buffer_address(), k]
    kern = ttnn.KernelDescriptor(kernel_source=str(KDIR / "slab_write.cpp"), core_ranges=crs, compile_time_args=ct,
                                 runtime_args=rt,
                                 config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                                          noc=ttnn.NOC.NOC_0))
    cbs = [_cb(crs, 0, S * pageb), _cb(crs, 1, offb)]
    ttnn.generic_op([src, offs, dst], ttnn.ProgramDescriptor(kernels=[kern], semaphores=[], cbs=cbs))
    return dst
