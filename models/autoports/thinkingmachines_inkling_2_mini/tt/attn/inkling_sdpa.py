# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Inkling paged SDPA decode (one row, or the R rows of one sequence in a spec-verify step) as one ``ttnn.generic_op``:
the qwen36 rows SDPA decode (attn_decode/rows_sdpa.py, description below) plus

  * Inkling's relative-position bias: per (row, head) query a bias over the key distance, (r_q @ proj)[q - k] for
    q - k < E (tt/relbias.py). Inputs ``rq`` [1, R, NH, 32] (each head's 16-wide r vector, zero-padded, pre-multiplied by
    tau on global layers past 128K) and ``table`` (relbias.build_tables for this layer, tilized [1, 32, 32, M*32]). Each
    band chunk computes BIAS = sum_j Rq_j @ T[phi_j][m_j] in the compute kernel and the QK matmul adds it, with the
    causal / window mask folded in; far chunks (distance >= E, global layers) are the stock path.
  * a sliding window W (0 = none): row j attends [cur_pos[j] + 1 - W, cur_pos[j]]; only chunks inside the rows' windows
    are read (the stock helper's sliding-window workload), the window edge is masked per row. Used with a ring KV cache
    whose page table maps logical block b to physical block b mod n_ring (the caller's page table): every position in
    a row's window maps to a live block, so the cache stays W + one block deep however long the sequence.

The scale is applied after the bias add (exp((QK + bias - max) * scale)), hence the table holds proj / scale.

--- rows SDPA decode (the base) ---
Paged SDPA decode for the R rows of ONE sequence (an MTP / DFlash verify step: rows at their own positions, row r
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

from models.autoports.thinkingmachines_inkling_2_mini.tt import relbias
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


def inkling_sdpa_decode(
    q,
    k_cache,
    v_cache,
    cur_pos,
    page_table,
    rq,
    table,
    scale,
    window,
    extent,
    k_chunk_size=0,
    out_memory_config=None,
    sink=None,
    pt_per_item=False,
    mi=1,
    table2=None,
    one_min=0,
):
    """-> [1, n_items * R, NH, HD] bf16, or [1, 1, n_items * R, NH * HD] when mi * R == 32 (freshly allocated; the
    caller frees it). window: 0 (global) or the sliding window; extent: rel_extent E (the table must be
    relbias.build_tables(proj, scale) for this E).

    cur_pos [R] (one item: decode / spec verify; the K span of each kv head is split over many cores) or [n_items, R]
    (prefill: n_items Q tiles of R consecutive rows each, q / rq / out rows stacked by item; one core per (item, kv
    head) unit, the units spread over the grid). Item i's rows are sequence rows i * R .. i * R + R - 1.

    Inkling-2: ``sink`` [1, n_items * R, NH, 32] bf16 TILE holds each row's per-head sink logit DIVIDED BY ``scale`` in
    every column (the kernel works on unscaled logits, see the table); a zero-value extra key per row and head.
    ``pt_per_item``: item i reads page-table row i (n_items independent sequences, multi-user decode) instead of row 0.
    ``mi`` (prefill): consecutive items per work unit (one sequence: with pt_per_item, groups of mi items must not
    straddle sequences); each K / V chunk is read once for the mi items' stacked Q tiles (long contexts: the K / V
    stream, not the compute, bounds the one-item units).
    ``table2`` (prefill, mi > 1): relbias.build_pair_tables of the table: rows 2p / 2p + 1 of an item (consecutive
    positions) share one Rq tile and one table tile (half the bias matmuls and table reads; bitwise).
    ``one_min`` (one item): a kv head with at most this many chunks runs on one core (as SDPA_MU_MIN).
    """
    n_items = int(cur_pos.shape[0]) if len(cur_pos.shape) == 2 else 1
    R = int(cur_pos.shape[-1])
    NH, HD = int(q.shape[2]), int(q.shape[3])
    if int(q.shape[1]) != n_items * R:
        raise ValueError(f"inkling_sdpa_decode: q {tuple(q.shape)} for {n_items} items x {R} rows")
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
    if tuple(rq.shape) != (1, n_items * R, NH, TILE) or rq.dtype != ttnn.bfloat16 or rq.layout != ttnn.TILE_LAYOUT:
        raise ValueError(
            f"inkling_sdpa_decode: rq must be [1, {n_items * R}, {NH}, 32] bf16 TILE, got {tuple(rq.shape)}"
        )
    M_tiles = relbias.n_tiles(extent)
    if tuple(table.shape) != (1, TILE, TILE, M_tiles * TILE) or table.dtype != ttnn.bfloat16:
        raise ValueError(f"inkling_sdpa_decode: table {tuple(table.shape)} for extent {extent}")
    if sink is None or tuple(sink.shape) != (1, n_items * R, NH, TILE) or sink.dtype != ttnn.bfloat16:
        raise ValueError(f"inkling_sdpa_decode: sink must be [1, {n_items * R}, {NH}, 32] bf16 TILE")
    if pt_per_item and int(page_table.shape[0]) < n_items:
        raise ValueError(f"inkling_sdpa_decode: pt_per_item needs {n_items} page-table rows, got {page_table.shape}")
    if cur_pos.dtype != ttnn.int32:
        raise ValueError(f"rows_sdpa_decode: cur_pos {tuple(cur_pos.shape)} {cur_pos.dtype} for {R} rows")
    if n_items > 1 and not k_chunk_size:
        # prefill: a core runs several units; the stock dynamic chunk size grows with each unit's position, and CB
        # blocks of different sizes on one core would misalign the circular buffers. One fixed chunk for all units.
        k_chunk_size = DST_TILES * TILE
    Sk_fixed = k_chunk_size // TILE if k_chunk_size else 0
    if k_chunk_size % TILE or Sk_fixed > DST_TILES:
        raise ValueError(f"rows_sdpa_decode: k_chunk_size {k_chunk_size} (multiple of 32, <= {DST_TILES * TILE})")
    G = NH // NKV
    DHt = HD // TILE
    mesh = q.device()
    if n_items == 1 or n_items % mi:
        mi = 1
    if table2 is not None and PAIR and mi > 1 and n_items > 1 and R % 2 == 0:
        if tuple(table2.shape) != (1, TILE, TILE, (M_tiles + 1) * TILE):
            raise ValueError(f"inkling_sdpa_decode: table2 {tuple(table2.shape)} for extent {extent}")
        table = table2
    tbl_l1 = None
    if table is table2 and TBL_L1 and table.memory_config().buffer_type == ttnn.BufferType.DRAM:
        # prefill: every unit reads (nearly) the whole bias table; from L1 banks instead of DRAM (bitwise). Only with
        # the pair tables: their smaller CBs (-160 KB per core) leave room for the copy (~20 KB per bank, freed
        # after the call) below the persistent L1 buffers
        table = tbl_l1 = ttnn.to_memory_config(table, ttnn.L1_MEMORY_CONFIG)
    if n_items > 1 and _PF_CACHE_ON:  # prefill: the program's static parts per shape (see _PF_CACHE)
        key = (
            id(mesh),
            n_items,
            R,
            NH,
            HD,
            NKV,
            block,
            str(k_cache.dtype),
            str(v_cache.dtype),
            window,
            extent,
            M_tiles,
            Sk_fixed,
            int(page_table.shape[-1]),
            float(scale),
            int(table.shape[-1]),
            repr(out_memory_config),
            bool(pt_per_item),
            mi,
            MU_SPLIT,
            MU_MIN,
            TRIM,
            BCAST,
            BALANCE,
            FIN2,
            FUSE,
            tuple(repr(t.memory_config()) for t in (q, k_cache, v_cache, cur_pos, page_table, rq, table, sink)),
        )
        st = _PF_CACHE.get(key)
        if st is None:
            st = _PF_CACHE[key] = _pf_static(
                q,
                k_cache,
                v_cache,
                cur_pos,
                page_table,
                rq,
                table,
                scale,
                window,
                extent,
                Sk_fixed,
                M_tiles,
                out_memory_config,
                key,
                sink,
                pt_per_item,
                mi,
            )
        out = _pf_launch(st, q, k_cache, v_cache, cur_pos, page_table, rq, table, sink)
        if tbl_l1 is not None:
            ttnn.deallocate(tbl_l1)
        return out
    out = _build_and_run(
        q,
        k_cache,
        v_cache,
        cur_pos,
        page_table,
        rq,
        table,
        scale,
        window,
        extent,
        Sk_fixed,
        M_tiles,
        out_memory_config,
        n_items,
        R,
        NH,
        HD,
        NKV,
        block,
        G,
        DHt,
        mesh,
        sink=sink,
        pt_per_item=pt_per_item,
        mi=mi,
        one_min=one_min,
    )
    if tbl_l1 is not None:
        ttnn.deallocate(tbl_l1)
    return out


def _build_and_run(
    q,
    k_cache,
    v_cache,
    cur_pos,
    page_table,
    rq,
    table,
    scale,
    window,
    extent,
    Sk_fixed,
    M_tiles,
    out_memory_config,
    n_items,
    R,
    NH,
    HD,
    NKV,
    block,
    G,
    DHt,
    mesh,
    static_only=False,
    sink=None,
    pt_per_item=False,
    mi=1,
    one_min=0,
):
    """The program for one call (static_only: its static parts, no launch; _pf_static)."""
    grid = mesh.compute_with_storage_grid_size()
    n_units = n_items // mi * NKV
    mu_min = 0
    if n_items == 1:  # decode / verify: groups = kv heads, cpk cores per group split the K span (tree reduction)
        cpk = min((grid.x * grid.y) // NKV, 1 << MAX_TREE_ROUNDS)  # cores per kv head
        if cpk < 1:
            raise ValueError(f"rows_sdpa_decode: {NKV} kv heads on a {grid.x}x{grid.y} grid")
        n_groups = NKV
        mu_min = one_min
    elif MU_SPLIT and pt_per_item and mi == 1 and 2 * n_units <= grid.x * grid.y:
        # multi-user decode with room for >= 2 cores per (user, kv head): split each unit's K span as for one item
        cpk = min((grid.x * grid.y) // n_units, 1 << MAX_TREE_ROUNDS)
        n_groups = n_units
        mu_min = MU_MIN
    else:  # prefill: one core per group, the units round-robin over the groups
        cpk = 1
        n_groups = min(grid.x * grid.y, n_units)
    n_cores = n_groups * cpk
    cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(n_cores)]  # row-major (stock op)
    noc = [mesh.worker_core_from_logical_core(c) for c in cores]
    cr = _range_set(cores)
    rounds = (cpk - 1).bit_length() if cpk > 1 else 0

    # mi items of R rows = one 32-row tile row: the writer assembles whole tiles of [1, 1, T, NH * HD] (no reshape after)
    out_shape = ttnn.Shape([1, 1, n_items * R, NH * HD] if mi * R == TILE else [1, n_items * R, NH, HD])
    out_mc = out_memory_config or (ttnn.L1_MEMORY_CONFIG if n_items == 1 else ttnn.DRAM_MEMORY_CONFIG)
    out = (
        None
        if static_only
        else ttnn.allocate_tensor_on_device(out_shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, out_mc)
    )

    Sk_cb = Sk_fixed or DST_TILES
    # one unit per core (decode): a core's last chunk reads / computes only its live key tiles (kernels: SDPA_TRIM)
    trim = TRIM and cpk > 1 and Sk_cb in (1, 2, 4, 8)
    # one unit per core (decode): core 0 reads every item's position / page-table prefix / Q / Rq / sink and multicasts
    # them to the program's cores (kernels: SDPA_BCAST); per item a slot of the scratch CB
    bcast = BCAST and cpk > 1 and mi == 1
    pt_pre = min(BCAST_PT, _align(int(page_table.shape[-1]) * 4))
    slot = _align(R * 4) + pt_pre + R * (DHt + 2) * 2048
    # (+ one tile past the slots: the writer's fully masked tile)
    scr_pages = -(-(n_items * slot + 128) // 2048) + 1 if bcast else R * (DHt + 2) + 1
    rects = _mcast_rects(noc, grid.x) if bcast else []
    bf, bt = ttnn.bfloat16, 2048
    kt, vt = _tile_bytes(k_cache.dtype), _tile_bytes(v_cache.dtype)
    pos_stick = _align(R * 4)
    pair = int(table.shape[-1]) == (M_tiles + 1) * TILE  # (the row-pair tables, see inkling_sdpa_decode)
    RP = R // 2 if pair else R
    defs = _mi_defines(mi, pos_stick, trim, bcast, pt_pre) + [("SDPA_MU_MIN", str(mu_min)), ("SDPA_PAIR", str(int(pair)))]
    if bcast:
        defs.append(("SDPA_NEG_OFF", str(_align(n_items * slot))))
    pt_stick = _align(int(page_table.shape[-1]) * 4)
    cbs = [
        _cb(cr, 0, bf, bt, mi * DHt),  # q (gathered; mi items' tile rows)
        _cb(cr, 1, k_cache.dtype, kt, Sk_cb * DHt * 2),  # K chunks, double-buffered
        _cb(cr, 2, v_cache.dtype, vt, Sk_cb * DHt * 2),  # V chunks
        _cb(cr, 3, bf, bt, Sk_cb * 2),  # per-row mask blocks
        _cb(cr, 4, bf, bt, mi * RP),  # Rq tiles (one per verify row; SDPA_PAIR: per row pair)
        _cb(cr, 11, bf, bt, RP * Sk_cb * 2),  # bias-table tiles, double-buffered
        _cb(cr, 13, bf, bt, mi * Sk_cb),  # bias (+ mask) block
        _cb(cr, 14, bf, bt, mi),  # Inkling-2: per-row sinks
        _cb(cr, 5, bf, bt, 1),  # reduce scaler
        _cb(cr, 6, bf, bt, 1),  # child max
        _cb(cr, 7, bf, bt, 1),  # child sum
        _cb(cr, 8, ttnn.uint32, pos_stick, mi),  # cur_pos (writer)
        _cb(cr, 9, ttnn.uint32, pt_stick, 1),  # page-table row
        _cb(cr, 10, bf, bt, scr_pages),  # q / rq / sink scratch (+ 64-byte alignment slack) or the broadcast slots
        _cb(cr, 12, bf, bt, 1),  # zero tile
        _cb(cr, 15, ttnn.uint32, pos_stick, mi),  # cur_pos (compute)
        _cb(cr, 16, bf, bt, DHt),  # O to the parent / from a child
        _cb(cr, 17, bf, bt, 1),
        _cb(cr, 18, bf, bt, 1),
        _cb(cr, 20, bf, bt, mi * DHt),  # root result
        _cb(cr, 21, bf, bt, mi),
        _cb(cr, 22, bf, bt, mi),
        _cb(cr, 23, bf, bt, DHt),
        _cb(cr, 24, bf, bt, mi * Sk_cb * (2 if trim else 1)),  # QK / P (trim: a smaller last block never wraps)
        _cb(cr, 25, bf, bt, mi * DHt),
        _cb(cr, 26, bf, bt, mi * DHt),
    ] + [_cb(cr, i, bf, bt, mi) for i in range(27, 32)]
    if rounds:
        cbs.append(_cb(cr, 19, bf, bt, rounds * (DHt + 2)))  # children's (l, m, o), one slot per round
    elif mi * R == TILE:
        cbs.append(_cb(cr, 19, bf, bt, G * DHt))  # the output tiles of a unit's G heads (assembled)

    rd_ct = [R, G, DHt, DHt, Sk_fixed, DST_TILES, NKV, block // TILE, cpk, pos_stick, n_cores, window, extent, M_tiles]
    rd_ct += [int(bool(pt_per_item))]
    for t in (q, k_cache, v_cache, cur_pos, page_table, rq, table, sink):
        rd_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
    wr_ct = [R, G, DHt, Sk_fixed, DST_TILES, cpk, 0, n_cores, MASK_NEG_BF16, window, NKV]
    if static_only:  # (the output's accessor args from a tensor of the same layout, allocated once and freed)
        o_ref = ttnn.allocate_tensor_on_device(out_shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, out_mc)
        wr_ct += ttnn.TensorAccessorArgs(o_ref).get_compile_time_args()
        ttnn.deallocate(o_ref)
    else:
        wr_ct += ttnn.TensorAccessorArgs(out).get_compile_time_args()
    wr_ct += [extent, M_tiles] + ttnn.TensorAccessorArgs(table).get_compile_time_args()
    cp_ct = [DHt, DHt, Sk_fixed, DST_TILES, cpk, R, struct.unpack("<I", struct.pack("<f", float(scale)))[0], DST_TILES]
    cp_ct += [window, extent, G]

    if static_only:
        per_core = []
        for i in range(n_cores):
            g, c = divmod(i, cpk)
            is_root, parent, send, children = tree_params(c, cpk)
            group = noc[g * cpk : (g + 1) * cpk]
            per_core.append(
                (
                    [pt_stick, g, c],
                    [g, c, int(is_root), parent, send]
                    + children
                    + [p.x for p in group]
                    + [p.y for p in group]
                    + [n_groups, n_units],
                    [c, int(is_root), parent] + children + [g, n_groups, n_units],
                    [len(rects)] + [v for r in rects for v in r],
                )
            )
        return dict(
            cores=cores,
            cr=cr,
            cbs=cbs,
            rd_ct=rd_ct,
            wr_ct=wr_ct,
            cp_ct=cp_ct,
            per_core=per_core,
            n_groups=n_groups,
            n_units=n_units,
            out_shape=out_shape,
            out_mc=out_mc,
            defs=defs,
        )
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
                rq.buffer_address(),
                table.buffer_address(),
                n_groups,
                n_units,
                sink.buffer_address(),
            ]
            + [len(rects)]
            + [v for r in rects for v in r]
        )
        wr_rt.append(
            [out.buffer_address(), g, c, int(is_root), parent, send]
            + children
            + [p.x for p in group]
            + [p.y for p in group]
            + [n_groups, n_units, table.buffer_address()]
        )
        cp_rt.append([c, int(is_root), parent] + children + [g, n_groups, n_units])


    def dm(source, ct, rt, riscv, noc_id):
        return ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / source),
            core_ranges=cr,
            compile_time_args=ct,
            runtime_args=_rt(cores, rt),
            config=ttnn.DataMovementConfigDescriptor(processor=riscv, noc=noc_id),
            defines=defs,
        )

    gran = min(DHt, DST_TILES)
    log2g = gran.bit_length() - 1
    if gran != 1 << log2g:
        gran, log2g = 1, 0
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=True
    )
    kernels = [
        dm("inkling_sdpa_reader.cpp", rd_ct, rd_rt, ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
        dm("inkling_sdpa_writer.cpp", wr_ct, wr_rt, ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "inkling_sdpa_compute.cpp"),
            core_ranges=cr,
            compile_time_args=cp_ct,
            defines=[
                ("EXP_APPROX_MODE", "0"),
                ("DHT_GRANULARITY", str(gran)),
                ("LOG2_DHT_GRANULARITY", str(log2g)),
                ("DYNAMIC_CHUNK_SIZE", "1"),
            ]
            + defs,
            runtime_args=_rt(cores, cp_rt),
            config=cfg,
        ),
    ]
    # 0: the tree reduction (writer), 1: core 0's broadcast of the positions / Q landed (reader, SDPA_BCAST)
    semaphores = [ttnn.SemaphoreDescriptor(i, ttnn.CoreType.WORKER, cr, 0) for i in range(2)]
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)
    ttnn.generic_op([q, k_cache, v_cache, cur_pos, page_table, rq, table, sink, out], program)
    return out


# INKLING_SDPA_PF_CACHE (default 1): a prefill call's program parts that do not change between calls of the same shape
# (cores, CBs, compile-time args, the per-core non-address arguments, the compute kernel's arguments) are built once per
# shape key and the program carries a stable custom hash (generic_op then skips hashing the descriptor). The same
# program as before: ~1.5 ms of Python per call (473 calls per 16K prompt) was its construction.
# I2M_SDPA_TRIM (default 1): skip the fully masked key tiles of a decode core's last chunk (bitwise)
TRIM = __import__("os").environ.get("I2M_SDPA_TRIM", "1") == "1"
# I2M_SDPA_BCAST (default 1): decode positions / Q read once by core 0 and multicast (bitwise)
BCAST = __import__("os").environ.get("I2M_SDPA_BCAST", "1") == "1"
BCAST_PT = 512  # page-table bytes per item in the broadcast (128 blocks: 8K positions at 64-row blocks)
# I2M_SDPA_BALANCE (default 0): with one chunk per core, the live key tiles split evenly over the chunks (same count;
# not bitwise: the partition boundaries move)
BALANCE = __import__("os").environ.get("I2M_SDPA_BALANCE", "0") == "1"
# I2M_SDPA_FIN2 (default 0): the sink and the normalization in 4 compute steps (O / (l + exp(s - m))) instead of 9 (not
# bitwise: fewer roundings)
FIN2 = __import__("os").environ.get("I2M_SDPA_FIN2", "0") == "1"
# I2M_SDPA_FUSE (default 1): fewer compute steps, the same arithmetic (bitwise): the running max / sum change CBs by
# swapping roles instead of copying, the reduction's O update and the finalize's sum / reciprocal / O scale keep their
# intermediates in DST (bf16 DST and CBs: a pack + unpack round trip is exact)
FUSE = __import__("os").environ.get("I2M_SDPA_FUSE", "1") == "1"
# I2M_SDPA_MU_SPLIT (default 1): multi-user decode (pt_per_item, few units) splits each (user, kv head) over cores
MU_SPLIT = __import__("os").environ.get("I2M_SDPA_MU_SPLIT", "1") == "1"
# I2M_SDPA_MU_MIN (default 4): a split unit with at most this many chunks (1024 positions) runs on one core, as unsplit
MU_MIN = int(__import__("os").environ.get("I2M_SDPA_MU_MIN", "4"))
# I2M_SDPA_TBL_L1 (default 1; with I2M_SDPA_PAIR): prefill reads the bias table from an L1-interleaved copy made per
# call and freed after it (DRAM-bound before: every unit reads ~the whole table; one die 2048 rows 3.7 -> 2.5 ms per
# call; bitwise)
TBL_L1 = __import__("os").environ.get("I2M_SDPA_TBL_L1", "1") == "1"
# I2M_SDPA_PAIR (default 1): prefill rows paired in the bias (table2): half the bias matmuls and table reads (bitwise)
PAIR = __import__("os").environ.get("I2M_SDPA_PAIR", "1") == "1"
_PF_CACHE_ON = __import__("os").environ.get("INKLING_SDPA_PF_CACHE", "1") == "1"
_PF_CACHE = {}


def _mi_defines(mi, pos_stick, trim=False, bcast=False, pt_pre=64):
    return [
        ("SDPA_MI", str(mi)),
        ("SDPA_POS_WORDS", str(pos_stick // 4)),
        ("SDPA_TRIM", str(int(trim))),
        ("SDPA_BCAST", str(int(bcast))),
        ("SDPA_BCAST_PT", str(pt_pre)),
        ("SDPA_BALANCE", str(int(trim and BALANCE))),
        ("SDPA_FIN2", str(int(FIN2))),
        ("SDPA_FUSE", str(int(FUSE))),
    ]


def _mcast_rects(noc, gx):
    """The program cores (the first len(noc) of the grid, row-major) as multicast rectangles from core 0:
    (x0, y0, x1, y1, destinations) in NOC coordinates; core 0 itself is never written (not counted)."""
    n = len(noc)
    full, rem = divmod(n, gx)
    rects = []
    if full:
        rects.append((0, full * gx - 1))
    if rem:
        rects.append((full * gx, n - 1))
    out = []
    for a, b in rects:
        dests = b - a + 1 - (1 if a == 0 else 0)
        if dests:
            out.append((noc[a].x, noc[a].y, noc[b].x, noc[b].y, dests))
    return out


def _stable_hash(key):
    import hashlib

    return int.from_bytes(hashlib.sha1(repr(key).encode()).digest()[:8], "little") & ((1 << 63) - 1)


def _pf_static(
    q,
    k_cache,
    v_cache,
    cur_pos,
    page_table,
    rq,
    table,
    scale,
    window,
    extent,
    Sk_fixed,
    M_tiles,
    out_memory_config,
    key,
    sink,
    pt_per_item,
    mi=1,
):
    n_items, R = int(cur_pos.shape[0]), int(cur_pos.shape[-1])
    NH, HD = int(q.shape[2]), int(q.shape[3])
    _, NKV, block, _ = (int(s) for s in k_cache.shape)
    st = _build_and_run(
        q,
        k_cache,
        v_cache,
        cur_pos,
        page_table,
        rq,
        table,
        scale,
        window,
        extent,
        Sk_fixed,
        M_tiles,
        out_memory_config,
        n_items,
        R,
        NH,
        HD,
        NKV,
        block,
        NH // NKV,
        HD // TILE,
        q.device(),
        static_only=True,
        sink=sink,
        pt_per_item=pt_per_item,
        mi=mi,
    )
    cores = st["cores"]
    st["cp_rtargs"] = _rt(cores, [pc[2] for pc in st["per_core"]])
    gran = min(HD // TILE, DST_TILES)
    log2g = gran.bit_length() - 1
    if gran != 1 << log2g:
        gran, log2g = 1, 0
    st["compute"] = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "inkling_sdpa_compute.cpp"),
        core_ranges=st["cr"],
        compile_time_args=st["cp_ct"],
        defines=[
            ("EXP_APPROX_MODE", "0"),
            ("DHT_GRANULARITY", str(gran)),
            ("LOG2_DHT_GRANULARITY", str(log2g)),
            ("DYNAMIC_CHUNK_SIZE", "1"),
        ]
        + st["defs"],
        runtime_args=st["cp_rtargs"],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=True
        ),
    )
    st["sem"] = [ttnn.SemaphoreDescriptor(i, ttnn.CoreType.WORKER, st["cr"], 0) for i in range(2)]
    st["hash"] = _stable_hash(("inkling_sdpa_pf", key, st["rd_ct"], st["wr_ct"], st["cp_ct"]))
    return st


def _pf_launch(st, q, k_cache, v_cache, cur_pos, page_table, rq, table, sink):
    mesh = q.device()
    out = ttnn.allocate_tensor_on_device(st["out_shape"], ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, st["out_mc"])
    a = [
        q.buffer_address(),
        k_cache.buffer_address(),
        v_cache.buffer_address(),
        cur_pos.buffer_address(),
        page_table.buffer_address(),
    ]
    rqa, ta, oa = rq.buffer_address(), table.buffer_address(), out.buffer_address()
    n_groups, n_units = st["n_groups"], st["n_units"]
    rd_rt = [a + pc[0] + [rqa, ta, n_groups, n_units, sink.buffer_address()] + pc[3] for pc in st["per_core"]]
    wr_rt = [[oa] + pc[1] + [ta] for pc in st["per_core"]]
    cores, cr = st["cores"], st["cr"]

    def dm(source, ct, rt, riscv, noc_id):
        return ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / source),
            core_ranges=cr,
            compile_time_args=ct,
            runtime_args=_rt(cores, rt),
            config=ttnn.DataMovementConfigDescriptor(processor=riscv, noc=noc_id),
            defines=st["defs"],
        )

    kernels = [
        dm("inkling_sdpa_reader.cpp", st["rd_ct"], rd_rt, ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
        dm("inkling_sdpa_writer.cpp", st["wr_ct"], wr_rt, ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
        st["compute"],
    ]
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=st["sem"], cbs=st["cbs"])
    program.custom_program_hash = st["hash"]
    ttnn.generic_op([q, k_cache, v_cache, cur_pos, page_table, rq, table, sink, out], program)
    return out
