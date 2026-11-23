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


def slab_write(src, dst, offs, n_slabs: int, slab_rows: int = 32, src_offs: bool = False):
    """dst rows [offs[k], offs[k] + slab_rows) = src rows [k slab_rows, (k + 1) slab_rows), k < n_slabs.
    Row-major src / dst (same row width / dtype): any offsets. Tiled src / dst (slab_rows = 32): whole tile rows,
    offsets multiples of 32. offs [1, 1, 1, >= n_slabs] uint32 RM on the device.
    src_offs=True: source slab k starts at row offs[n_slabs + k] instead (offs [1, 1, 1, >= 2 n_slabs]): a gather /
    scatter between any rows (src and dst may be the same tensor when no slab's source overlaps another's destination)."""
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
    ct = [S, pageb, offb, 0, 1, n_slabs, nc, int(tiled), int(bool(src_offs))] + acc(src) + acc(dst) + acc(offs)
    rt = ttnn.RuntimeArgs()
    for k, c in enumerate(cores):
        rt[c.x][c.y] = [src.buffer_address(), dst.buffer_address(), offs.buffer_address(), k]
    kern = ttnn.KernelDescriptor(kernel_source=str(KDIR / "slab_write.cpp"), core_ranges=crs, compile_time_args=ct,
                                 runtime_args=rt,
                                 config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                                          noc=ttnn.NOC.NOC_0))
    cbs = [_cb(crs, 0, S * pageb), _cb(crs, 1, offb)]
    ios = [offs, dst] if src is dst else [src, offs, dst]
    ttnn.generic_op(ios, ttnn.ProgramDescriptor(kernels=[kern], semaphores=[], cbs=cbs))
    return dst


def kv_rows_write(a, b, dst, offs, n_rows: int, zero_bytes: int):
    """dst rows offs[k] (k < n_rows) = row k of [a | b | zero_bytes zeros]: a, b bf16 TILE tensors of one tile row (L1:
    the kernel reads 32-byte face rows), dst row-major bf16 of 64 (a + b tiles) + zero_bytes bytes per row. One program
    for concat + to_layout + slab_write (HY4_KVROW_FUSED; the same bytes)."""
    assert a.layout == ttnn.TILE_LAYOUT and b.layout == ttnn.TILE_LAYOUT and dst.layout == ttnn.ROW_MAJOR_LAYOUT
    A, B = int(a.shape[-1]) // 32, int(b.shape[-1]) // 32
    rowb = int(dst.shape[-1]) * dst.element_size()
    assert rowb == 64 * (A + B) + zero_bytes and zero_bytes % 4 == 0
    offb = -(-(int(offs.shape[-1]) * 4) // 32) * 32
    grid = dst.device().compute_with_storage_grid_size()
    nc = min(n_rows, grid.x * grid.y)
    cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(nc)]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    acc = lambda t: ttnn.TensorAccessorArgs(t).get_compile_time_args()  # noqa: E731
    ct = [A, B, zero_bytes, rowb, offb, 0, 1, n_rows, nc] + acc(a) + acc(b) + acc(dst) + acc(offs)
    rt = ttnn.RuntimeArgs()
    for k, c in enumerate(cores):
        rt[c.x][c.y] = [a.buffer_address(), b.buffer_address(), dst.buffer_address(), offs.buffer_address(), k]
    kern = ttnn.KernelDescriptor(kernel_source=str(KDIR / "kv_rows.cpp"), core_ranges=crs, compile_time_args=ct,
                                 runtime_args=rt,
                                 config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                                          noc=ttnn.NOC.NOC_0))
    cbs = [_cb(crs, 0, -(-rowb // 32) * 32), _cb(crs, 1, offb)]
    ttnn.generic_op([a, b, offs, dst], ttnn.ProgramDescriptor(kernels=[kern], semaphores=[], cbs=cbs))
    return dst
