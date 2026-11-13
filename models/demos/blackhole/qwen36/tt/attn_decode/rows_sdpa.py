# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Paged SDPA decode for the R rows of ONE sequence (an MTP / DFlash verify step: rows at their own positions, row r
attending causally to positions 0 .. cur_pos[r]) that reads every K / V chunk ONCE for all rows, as one
``ttnn.generic_op`` program.

The stock ``paged_scaled_dot_product_attention_decode`` with the rows as R users reads the whole paged KV cache R times
(MiniMax-M2.7 per die, 12 q / 2 kv heads, bfp8 KV: 1 vs 4 users 85 vs 267 us at 40K context). Here the R rows' query
heads of one kv head (R x G <= 32, G = q heads per kv head) share one 32-row Q tile (tile row r*G+i = row r's head
g*G+i), and the stock op's flash-decode program runs on it: the KV span [0, max_r cur_pos[r]] split in chunks over the
cores of each kv head (the stock op's split, cores per kv head = grid / NKV up to 64), online softmax per core, the stock
tree reduction. The stock op masks the last chunk at one position; here each chunk whose last position is past the
oldest row limit carries a per-row mask (tile row rho masks the positions past its row's limit), so rows straddling a
chunk or a core boundary each keep their own causal limit (kernels/rows_sdpa_*.cpp).

Inputs: q [1, R, NH, HD] bf16 TILE (DRAM interleaved: the prep op's q, heads as tile rows, NH <= 32); the paged caches
[blocks, NKV, block, HD] TILE; cur_pos [R] int32 ROW_MAJOR (-1 = skipped row: its output rows are left unwritten, as
the stock op leaves a skipped user's); page_table [rows, max_blocks] int32 ROW_MAJOR, every row the same sequence (row 0
is read). Output [1, R, NH, HD] bf16 TILE in L1 (rows past NH of each tile unwritten, as the stock op's). The data
formats, math fidelity (HiFi2), approximations and chunk sizes are the stock op's defaults; k_chunk_size 0 = the stock
dynamic chunk size.
"""

from __future__ import annotations

import struct
from pathlib import Path

import ttnn
from models.demos.blackhole.qwen36.tt.gdn_decode.ops import TILE, _cb, _range_set, _rt

KERNEL_DIR = Path(__file__).parent / "kernels"
MAX_TREE_ROUNDS = 6  # the stock op's MAX_TREE_REDUCTION_ROUNDS (64 cores per kv head)
DST_TILES = 8  # bf16 DST (fp32_dest_acc_en off, the stock op's default): the largest K chunk in tiles
MASK_NEG_BF16 = 0xF14A  # -1e30: finite, see kernels/rows_sdpa_writer.cpp
U32_MAX = 0xFFFFFFFF


def _tile_bytes(dtype):
    return {ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048, ttnn.bfloat4_b: 576}[dtype]


def _align(n, a=64):
    return (n + a - 1) // a * a


def tree_params(core_id, n):
    """The stock op's get_tree_reduction_params (sdpa_decode_device_operation.hpp): (is_root, parent, send_at_round,
    children_per_round[MAX_TREE_ROUNDS]) of core core_id among n (core 0 = root, holding the last chunks)."""
    children = [U32_MAX] * MAX_TREE_ROUNDS
    if n <= 1:
        return True, U32_MAX, U32_MAX, children

    def trailing_ones(x):
        t = 0
        while x & 1:
            x >>= 1
            t += 1
        return t

    vid = (n - 1) - core_id
    rounds = (n - 1).bit_length()
    is_root = core_id == 0
    for r in range(rounds):
        mask = (2 << r) - 1
        if vid & mask == mask:
            child_vid = vid - (1 << r)
            if child_vid < n:
                children[r] = (n - 1) - child_vid
    parent, send = U32_MAX, U32_MAX
    if not is_root:
        t = trailing_ones(vid)
        parent_vid = vid + (1 << t)
        parent = (n - 1) - parent_vid if parent_vid < n else 0
        send = t
    else:
        for c in range(1, n):
            cv = (n - 1) - c
            t = trailing_ones(cv)
            if cv + (1 << t) >= n and children[t] == U32_MAX:
                children[t] = c
    return is_root, parent, send, children


def rows_sdpa_decode(q, k_cache, v_cache, cur_pos, page_table, scale, k_chunk_size=0):
    """-> [1, R, NH, HD] bf16 L1 (freshly allocated; the caller frees it)."""
    R, NH, HD = int(q.shape[1]), int(q.shape[2]), int(q.shape[3])
    _, NKV, block, HDk = (int(s) for s in k_cache.shape)
    if tuple(v_cache.shape) != tuple(k_cache.shape) or HDk != HD or HD % TILE:
        raise ValueError(f"rows_sdpa_decode: q {tuple(q.shape)}, K {tuple(k_cache.shape)}, V {tuple(v_cache.shape)}")
    if NH % NKV or NH > TILE or R * (NH // NKV) > TILE or block % TILE:
        raise ValueError(
            f"rows_sdpa_decode: {R} rows x {NH // max(NKV, 1)} q heads per kv head must fit one tile "
            f"(NH {NH}, NKV {NKV}, block {block})"
        )
    if q.dtype != ttnn.bfloat16 or q.layout != ttnn.TILE_LAYOUT or q.is_sharded():
        raise ValueError(f"rows_sdpa_decode: q must be interleaved bf16 TILE, got {q.dtype} {q.layout}")
    if int(cur_pos.shape[-1]) != R or cur_pos.dtype != ttnn.int32:
        raise ValueError(f"rows_sdpa_decode: cur_pos {tuple(cur_pos.shape)} {cur_pos.dtype} for {R} rows")
    Sk_fixed = k_chunk_size // TILE if k_chunk_size else 0
    if k_chunk_size % TILE or Sk_fixed > DST_TILES:
        raise ValueError(f"rows_sdpa_decode: k_chunk_size {k_chunk_size} (multiple of 32, <= {DST_TILES * TILE})")
    G = NH // NKV
    DHt = HD // TILE
    mesh = q.device()
    grid = mesh.compute_with_storage_grid_size()
    cpk = min((grid.x * grid.y) // NKV, 1 << MAX_TREE_ROUNDS)  # cores per kv head
    if cpk < 1:
        raise ValueError(f"rows_sdpa_decode: {NKV} kv heads on a {grid.x}x{grid.y} grid")
    n_cores = NKV * cpk
    cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(n_cores)]  # row-major (stock op)
    noc = [mesh.worker_core_from_logical_core(c) for c in cores]
    cr = _range_set(cores)
    rounds = (cpk - 1).bit_length() if cpk > 1 else 0

    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, R, NH, HD]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, ttnn.L1_MEMORY_CONFIG
    )

    Sk_cb = Sk_fixed or DST_TILES
    bf, bt = ttnn.bfloat16, 2048
    kt, vt = _tile_bytes(k_cache.dtype), _tile_bytes(v_cache.dtype)
    pos_stick = _align(R * 4)
    pt_stick = _align(int(page_table.shape[-1]) * 4)
    cbs = [
        _cb(cr, 0, bf, bt, DHt),  # q (gathered)
        _cb(cr, 1, k_cache.dtype, kt, Sk_cb * DHt * 2),  # K chunks, double-buffered
        _cb(cr, 2, v_cache.dtype, vt, Sk_cb * DHt * 2),  # V chunks
        _cb(cr, 3, bf, bt, Sk_cb * 2),  # per-row mask blocks
        _cb(cr, 5, bf, bt, 1),  # reduce scaler
        _cb(cr, 6, bf, bt, 1),  # child max
        _cb(cr, 7, bf, bt, 1),  # child sum
        _cb(cr, 8, ttnn.uint32, pos_stick, 1),  # cur_pos (writer)
        _cb(cr, 9, ttnn.uint32, pt_stick, 1),  # page-table row
        _cb(cr, 10, bf, bt, R * DHt + 1),  # q scratch (every row's tile row; + 64-byte alignment slack)
        _cb(cr, 12, bf, bt, 1),  # zero tile
        _cb(cr, 15, ttnn.uint32, pos_stick, 1),  # cur_pos (compute)
        _cb(cr, 16, bf, bt, DHt),  # O to the parent / from a child
        _cb(cr, 17, bf, bt, 1),
        _cb(cr, 18, bf, bt, 1),
        _cb(cr, 20, bf, bt, DHt),  # root result
        _cb(cr, 21, bf, bt, 1),
        _cb(cr, 22, bf, bt, 1),
        _cb(cr, 23, bf, bt, DHt),
        _cb(cr, 24, bf, bt, Sk_cb),  # QK / P
        _cb(cr, 25, bf, bt, DHt),
        _cb(cr, 26, bf, bt, DHt),
    ] + [_cb(cr, i, bf, bt, 1) for i in range(27, 32)]
    if rounds:
        cbs.append(_cb(cr, 19, bf, bt, rounds * (DHt + 2)))  # children's (l, m, o), one slot per round

    rd_ct = [R, G, DHt, DHt, Sk_fixed, DST_TILES, NKV, block // TILE, cpk, pos_stick, n_cores]
    for t in (q, k_cache, v_cache, cur_pos, page_table):
        rd_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
    wr_ct = [R, G, DHt, Sk_fixed, DST_TILES, cpk, 0, n_cores, MASK_NEG_BF16]
    wr_ct += ttnn.TensorAccessorArgs(out).get_compile_time_args()
    cp_ct = [DHt, DHt, Sk_fixed, DST_TILES, cpk, R, struct.unpack("<I", struct.pack("<f", float(scale)))[0], DST_TILES]

    rd_rt, wr_rt, cp_rt = [], [], []
    for i in range(n_cores):
        g, c = divmod(i, cpk)
        is_root, parent, send, children = tree_params(c, cpk)
        group = noc[g * cpk : (g + 1) * cpk]
        rd_rt.append(
            [
                q.buffer_address(),
                k_cache.buffer_address(),
                v_cache.buffer_address(),
                cur_pos.buffer_address(),
                page_table.buffer_address(),
                pt_stick,
                g,
                c,
            ]
        )
        wr_rt.append(
            [out.buffer_address(), g, c, int(is_root), parent, send]
            + children
            + [p.x for p in group]
            + [p.y for p in group]
        )
        cp_rt.append([c, int(is_root), parent] + children)

    def dm(source, ct, rt, riscv, noc_id):
        return ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / source),
            core_ranges=cr,
            compile_time_args=ct,
            runtime_args=_rt(cores, rt),
            config=ttnn.DataMovementConfigDescriptor(processor=riscv, noc=noc_id),
        )

    gran = min(DHt, DST_TILES)
    log2g = gran.bit_length() - 1
    if gran != 1 << log2g:
        gran, log2g = 1, 0
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=True
    )
    kernels = [
        dm("rows_sdpa_reader.cpp", rd_ct, rd_rt, ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
        dm("rows_sdpa_writer.cpp", wr_ct, wr_rt, ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "rows_sdpa_compute.cpp"),
            core_ranges=cr,
            compile_time_args=cp_ct,
            defines=[
                ("EXP_APPROX_MODE", "0"),
                ("DHT_GRANULARITY", str(gran)),
                ("LOG2_DHT_GRANULARITY", str(log2g)),
                ("DYNAMIC_CHUNK_SIZE", "1"),
            ],
            runtime_args=_rt(cores, cp_rt),
            config=cfg,
        ),
    ]
    semaphores = [ttnn.SemaphoreDescriptor(0, ttnn.CoreType.WORKER, cr, 0)]
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)
    ttnn.generic_op([q, k_cache, v_cache, cur_pos, page_table, out], program)
    return out
