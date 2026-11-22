# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Multi-core drop-in for ttnn.experimental.moe_sorted_dispatch (kernels/dispatch_mc.cpp), as one ttnn.generic_op.

The single-core op walks all T*k (token, slot) pairs twice on one RISC-V: ~1 ms per MoE layer at a 2048-token
prefill (15 % of Ornith-35B's layer). Here N cores (one or more 32-token tiles each) count and scatter their own
pairs; a leader core lays out the expert-sorted rows. The outputs are bit-identical to the single-core op:
    src_tok [1, Rd] uint32, weights [1, Rd] bf16, group_ids [1, G] uint16, undispatch [1, T*k] uint32 (ROW_MAJOR).
"""

from __future__ import annotations

import os
from pathlib import Path

import ttnn

KERNEL = str(Path(__file__).parent / "kernels" / "dispatch_mc.cpp")
MAX_CORES = int(os.environ.get("QWEN35_DISPATCH_MC_CORES", "64"))
L1_BUDGET = 1100 * 1024
ZERO_BYTES = 2048


def _a16(n: int) -> int:
    return (n + 15) // 16 * 16


def _a64(n: int) -> int:
    return (n + 63) // 64 * 64


def _cb(cores, idx, nbytes, fmt=ttnn.uint32):
    n = _a64(max(nbytes, 64))  # 64-byte multiples: every CB base stays DRAM-write aligned
    return ttnn.CBDescriptor(
        total_size=n, core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=n)],
    )


def fits(T: int, E: int, k: int, Rd: int, gs: int = 32, max_cores: int = MAX_CORES) -> bool:
    return _plan(T, E, k, Rd, gs, max_cores, 1 << 30)[-1] <= L1_BUDGET


def _plan(T, E, k, Rd, gs, max_cores, grid_n):
    num_tiles = -(-T // 32)
    N = max(1, min(num_tiles, max_cores, grid_n))
    tpc = -(-num_tiles // N)
    epc = -(-(-(-E // N)) // 4) * 4  # experts per collector: a multiple of 4 (16-byte slices)
    nc = -(-E // epc)
    P = tpc * 32 * k
    ents = min(E, P)
    stg_src = _a16(4 * P + 32 * ents + 16)
    stg_w = _a16(2 * P + 32 * ents + 16)
    G = Rd // gs
    sizes = dict(inp=tpc * (4096 + 2048), und=P * 4, vec=4 * nc * epc * 4, cntT=N * epc * 4, src=Rd * 4, w=Rd * 2,
                 gid=G * 2, scr=4 * E * 4, stg=stg_src + stg_w, zero=ZERO_BYTES)
    return num_tiles, N, tpc, nc, epc, stg_src, sizes, sum(_a64(v) for v in sizes.values())


def dispatch_mc(ids, scores, E: int, k: int, gs: int, Rd: int, pad_group_id: int = 0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, max_cores: int = MAX_CORES):
    dev = ids.device()
    T = int(ids.shape[-2])
    assert int(ids.shape[-1]) == k and ids.layout == ttnn.TILE_LAYOUT and scores.layout == ttnn.TILE_LAYOUT
    grid = dev.compute_with_storage_grid_size()
    num_tiles, N, tpc, nc, epc, stg_src, sizes, total = _plan(T, E, k, Rd, gs, max_cores, grid.x * grid.y)
    assert total <= L1_BUDGET, f"dispatch_mc: {total} B of L1 per core for T={T}"
    G = Rd // gs
    ids_elem = 2 if ids.dtype == ttnn.uint16 else 4
    ids_page = 1024 * ids_elem
    sc_page = 2048

    rm = ttnn.ROW_MAJOR_LAYOUT
    alloc = lambda n, dt: ttnn.allocate_tensor_on_device(ttnn.Shape([1, n]), dt, rm, dev, memory_config)
    src_tok, w, gid, und = alloc(Rd, ttnn.uint32), alloc(Rd, ttnn.bfloat16), alloc(G, ttnn.uint16), alloc(T * k, ttnn.uint32)

    logical = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(N)]
    phys = [dev.worker_core_from_logical_core(c) for c in logical]
    rows = {}
    for c in logical:
        rows.setdefault(c.y, []).append(c.x)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(min(xs), y), ttnn.CoreCoord(max(xs), y))
                               for y, xs in sorted(rows.items())])
    order = ["inp", "und", "vec", "cntT", "src", "w", "gid", "scr", "stg", "zero"]
    cbs = [_cb(cores, i, sizes[n]) for i, n in enumerate(order)]

    ct = [T, k, E, gs, Rd, G, ids_page, sc_page, ids_elem, pad_group_id, N, nc, epc, tpc] + list(range(len(order)))
    ct += [stg_src, ZERO_BYTES]
    for t in (ids, scores, src_tok, w, gid, und):
        ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
    base, rem = divmod(num_tiles, N)
    addrs = [t.buffer_address() for t in (ids, scores, src_tok, w, gid, und)]
    rt = ttnn.RuntimeArgs()
    t0 = 0
    for i, c in enumerate(logical):
        n = base + (1 if i < rem else 0)
        rt[c.x][c.y] = addrs + [i, t0, n] + [v for p in phys for v in (p.x, p.y)]
        t0 += n
    kernel = ttnn.KernelDescriptor(
        kernel_source=KERNEL, core_ranges=cores, compile_time_args=ct, runtime_args=rt,
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
    )
    sems = [ttnn.SemaphoreDescriptor(i, ttnn.CoreType.WORKER, cores, 0) for i in range(5)]
    ttnn.generic_op([ids, scores, src_tok, w, gid, und], ttnn.ProgramDescriptor(kernels=[kernel], semaphores=sems, cbs=cbs))
    return src_tok, w, gid, und
