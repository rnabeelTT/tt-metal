// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Shared-KV rows SDPA decode, reader (attn_decode/rows_sdpa.py). The stock paged SDPA decode reader
// (ttnn/.../sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp) cut to the paged, causal, DRAM-Q case, with the
// R verify rows of ONE sequence as one "user": the Q tile of kv head g holds row r's query heads g*G .. g*G+G-1 in tile
// rows r*G .. r*G+G-1 (rows past R*G zero), so each K / V chunk is read once for all rows.
//
// The chunk split follows the newest attended position (max over the rows); the per-row causal limits are applied by
// masks the writer builds (rows_sdpa_common.hpp).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp"
#include "ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp"
#include "inkling_sdpa_common.hpp"
#include "tools/profiler/kernel_profiler.hpp"

#ifndef SDPA_PAIR
#define SDPA_PAIR 0  // prefill: rows 2p / 2p + 1 share an Rq tile (r in columns 0-15 / 16-31)
#endif
#ifndef SDPA_MI
#define SDPA_MI 1  // prefill: consecutive items per unit (their Q tiles stacked; see the compute)
#endif
#ifndef SDPA_BCAST
#define SDPA_BCAST 0  // decode (one unit per core): program core 0 reads the positions / Q for all (below)
#endif
#ifndef SDPA_BCAST_PT
#define SDPA_BCAST_PT 64  // SDPA_BCAST: page-table bytes per item broadcast with the positions
#endif

void kernel_main() {
    Noc noc;

    constexpr uint32_t R = get_compile_time_arg_val(0);           // verify rows (users of the q tensor)
    constexpr uint32_t G = get_compile_time_arg_val(1);           // q heads per kv head
    constexpr uint32_t DHt = get_compile_time_arg_val(2);         // head dim in tiles
    constexpr uint32_t vDHt = get_compile_time_arg_val(3);        // V head dim in tiles
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(4);  // 0: dynamic (stock k_chunk_size 0)
    constexpr uint32_t max_dynamic_chunk_size = get_compile_time_arg_val(5);
    constexpr uint32_t num_kv_heads = get_compile_time_arg_val(6);
    constexpr uint32_t block_size_t = get_compile_time_arg_val(7);
    constexpr uint32_t num_cores_per_head = get_compile_time_arg_val(8);
    constexpr uint32_t pos_stick_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t num_cores = get_compile_time_arg_val(10);
    constexpr uint32_t W = get_compile_time_arg_val(11);        // sliding window (0 = global layer)
    constexpr uint32_t E = get_compile_time_arg_val(12);        // rel_extent: bias band
    constexpr uint32_t M_tiles = get_compile_time_arg_val(13);  // bias-table tiles per phase (E / 32 + 1)
    // Inkling-2: 1 = each item is its own sequence (multi-user decode): page-table row `item` instead of row 0
    constexpr uint32_t PT_PER_ITEM = get_compile_time_arg_val(14);
    constexpr auto q_args = TensorAccessorArgs<15>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto page_table_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    constexpr auto rq_args = TensorAccessorArgs<page_table_args.next_compile_time_args_offset()>();
    constexpr auto tbl_args = TensorAccessorArgs<rq_args.next_compile_time_args_offset()>();
    constexpr auto sink_args = TensorAccessorArgs<tbl_args.next_compile_time_args_offset()>();
    static_assert(R * G <= 32, "the rows' query heads of one kv head share one tile");
    constexpr uint32_t MI = SDPA_MI;
    constexpr uint32_t RP = SDPA_PAIR ? R / 2 : R;

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;
    constexpr uint32_t cb_writer_cur_pos = tt::CBIndex::c_8;
    constexpr uint32_t cb_id_page_table = tt::CBIndex::c_9;
    constexpr uint32_t cb_q_scratch = tt::CBIndex::c_10;
    constexpr uint32_t cb_compute_cur_pos = tt::CBIndex::c_15;
    constexpr uint32_t cb_rq = tt::CBIndex::c_4;    // R tiles: row j's G query heads' r (16 cols) at tile rows j*G..
    constexpr uint32_t cb_tbl = tt::CBIndex::c_11;  // per band chunk: R x Sk bias-table tiles (row-major [j][t])
    constexpr uint32_t cb_sink = tt::CBIndex::c_14;  // Inkling-2: 1 tile, row r*G+i = row r's head g*G+i sink / scale

    uint32_t arg_idx = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t k_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t v_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t pos_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t page_table_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t page_table_page_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t group = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_num_in_reduce = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t rq_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t tbl_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t n_groups = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t n_units = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sink_addr = get_arg_val<uint32_t>(arg_idx++);
    // the page-table row (an L1 buffer: the CB is never consumed), read per unit up to the blocks the unit attends (a
    // 1M-token row is 64 KB: reading it whole on every core cost ~5 ms per decode step)
    CircularBuffer cb_page_table(cb_id_page_table);
    cb_page_table.reserve_back(1);
    volatile tt_l1_ptr uint32_t* const pt_l1 =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_page_table.get_write_ptr());
    cb_page_table.push_back(1);
    volatile tt_l1_ptr uint32_t* page_table_ptr = pt_l1;  // (SDPA_BCAST: the broadcast prefix when it suffices)
    const auto pt_reader = TensorAccessor(page_table_args, page_table_addr, page_table_page_size);
    uint32_t pt_item = 0xFFFFFFFF, pt_bytes = 0;  // the row in pt_l1 and how much of it

    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);
    const uint8_t noc_id = noc.get_noc_id();
    const uint32_t my_noc_x = my_x[noc_id];
    const uint32_t my_noc_y = my_y[noc_id];
    UnicastEndpoint self;
    const auto pos_reader = TensorAccessor(pos_args, pos_addr);
    const auto q_reader = TensorAccessor(q_args, q_addr, q_tile_bytes);
    const auto rq_reader = TensorAccessor(rq_args, rq_addr, q_tile_bytes);
    const auto sink_reader = TensorAccessor(sink_args, sink_addr, q_tile_bytes);
    CircularBuffer cb_scr(cb_q_scratch);

    // SDPA_BCAST (decode: one unit per core, cpk cores per unit): program core 0 reads every item's position stick,
    // page-table prefix and whole Q / Rq / sink tiles in ONE DRAM round trip and multicasts them (then semaphore 1) to
    // every program core: before, ~100 cores each read the same 64-byte position stick (and the busy ones the same Q
    // rows) and serialized at those DRAM banks (~3 us + ~2 us). Per item a slot of the scratch CB: [position stick |
    // page-table prefix | Q R x DHt tiles | Rq R tiles | sink R tiles]
    constexpr uint32_t SLOT_TILES_OFF = pos_stick_bytes + SDPA_BCAST_PT;
    constexpr uint32_t SLOT = SLOT_TILES_OFF + R * (DHt + 2) * q_tile_bytes;
    uint32_t stage = 0;
    if constexpr (SDPA_BCAST) {
        cb_scr.reserve_back(1);
        stage = (cb_scr.get_write_ptr() + 63) & ~63u;
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
        DeviceZoneScopedN("S_R_BC");
        if (group == 0 && core_num_in_reduce == 0) {
            const uint32_t n_items = n_units / num_kv_heads;
            for (uint32_t k = 0; k < n_items; ++k) {
                const uint32_t slot = stage + k * SLOT;
                noc_async_read(pos_reader.get_noc_addr(k), slot, pos_stick_bytes);
                noc_async_read(
                    pt_reader.get_noc_addr(PT_PER_ITEM ? k : 0), slot + pos_stick_bytes,
                    SDPA_BCAST_PT < page_table_page_size ? SDPA_BCAST_PT : page_table_page_size);
                uint32_t dst = slot + SLOT_TILES_OFF;
                for (uint32_t t = 0; t < R * DHt; ++t, dst += q_tile_bytes) {
                    noc_async_read(q_reader.get_noc_addr(k * R * DHt + t), dst, q_tile_bytes);
                }
                for (uint32_t t = 0; t < R; ++t) {
                    noc_async_read(rq_reader.get_noc_addr(k * R + t), dst + t * q_tile_bytes, q_tile_bytes);
                    noc_async_read(sink_reader.get_noc_addr(k * R + t), dst + (R + t) * q_tile_bytes, q_tile_bytes);
                }
            }
            noc_async_read_barrier();
            // multicast rectangles of the program cores (count, then x0 y0 x1 y1 dests each; self is never written)
            const uint32_t n_rect = get_arg_val<uint32_t>(13);
            *sem = 1;
            for (uint32_t i = 0; i < n_rect; ++i) {
                const uint32_t b = 14 + 5 * i;
                const uint32_t x0 = get_arg_val<uint32_t>(b), y0 = get_arg_val<uint32_t>(b + 1);
                const uint32_t x1 = get_arg_val<uint32_t>(b + 2), y1 = get_arg_val<uint32_t>(b + 3);
                const uint32_t dests = get_arg_val<uint32_t>(b + 4);
                noc_async_write_multicast(
                    stage, get_noc_multicast_addr(x0, y0, x1, y1, stage), n_items * SLOT, dests, true);
                noc_semaphore_set_multicast(
                    get_semaphore(1), get_noc_multicast_addr(x0, y0, x1, y1, get_semaphore(1)), dests);
            }
            noc_async_write_barrier();
        } else {
            while (*sem == 0) {
                invalidate_l1_cache();
            }
        }
    }

    // Work units: unit = item * num_kv_heads + kv head. A unit is one 32-row Q tile (item = R consecutive rows of the
    // sequence x the G query heads of one kv head), computed by the num_cores_per_head cores of a group (decode: one
    // item, groups = kv heads, cores split the K span + tree reduction; prefill: one core per group, the items spread
    // over the groups round-robin, no reduction). Positions page `item` = that item's R rows; q / rq / out row r of
    // item i = sequence row i * R + r.
    for (uint32_t unit = group; unit < n_units; unit += n_groups) {
    const uint32_t cur_head = unit % num_kv_heads;
    const uint32_t item = (unit / num_kv_heads) * MI;  // the unit's first item (items item .. item + MI - 1)
    const uint32_t r0 = cur_head * G, r1 = cur_head * G + G;  // this kv head's q heads: tile rows [r0, r1)
    const uint32_t pt_row = PT_PER_ITEM ? item : 0;
    // tile rows [r0, r1) of a DRAM tile -> the same in-tile offsets of an L1 tile (whole 64-byte row pairs, so source
    // and destination agree mod 64)
    auto gather_rows = [&](uint64_t src, uint32_t dst) {
        for (uint32_t top = 0; top < 32; top += 16) {  // face rows 0-15 (faces 0, 1), 16-31 (faces 2, 3)
            const uint32_t a = r0 > top ? r0 : top;
            const uint32_t b = r1 < top + 16 ? r1 : top + 16;
            if (a >= b) {
                continue;
            }
            const uint32_t a2 = (a - top) & ~1u, b2 = ((b - top) + 1) & ~1u;  // row pairs within the face
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t off = (top >> 4) * 1024 + half * 512 + a2 * 32;
                noc_async_read(src + off, dst + off, (b2 - a2) * 32);
            }
        }
    };
    // rows [r0, r1) of a source tile, columns 0-15 -> rows (j / 2)*G .. of a destination tile, columns 16-31 for odd j
    // (SDPA_PAIR: the Rq tiles)
    auto place_pair = [&](uint32_t src_tile, uint32_t dst_tile, uint32_t j) {
        rows_sdpa_row_runs(r0, j * G, G, [&](uint32_t s, uint32_t t, uint32_t k) {
            noc.async_read(
                self,
                CoreLocalMem<uint32_t>(dst_tile + rows_sdpa_row_off(t) + (j & 1) * 512),
                k * 32,
                {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = src_tile + rows_sdpa_row_off(s)},
                {});
        });
    };
    // rows [r0, r1) of a source tile -> rows j*G .. j*G+G-1 of a destination tile (two 32-byte face rows per row)
    auto place_rows = [&](uint32_t src_tile, uint32_t dst_tile, uint32_t j) {
        rows_sdpa_row_runs(r0, j * G, G, [&](uint32_t s, uint32_t t, uint32_t k) {
            for (uint32_t half = 0; half < 2; ++half) {
                noc.async_read(
                    self,
                    CoreLocalMem<uint32_t>(dst_tile + rows_sdpa_row_off(t) + half * 512),
                    k * 32,
                    {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = src_tile + rows_sdpa_row_off(s) + half * 512},
                    {});
            }
        });
    };

    CircularBuffer cb_writer(cb_writer_cur_pos);
    CircularBuffer cb_compute(cb_compute_cur_pos);
    CircularBuffer cb_q(cb_q_in);
    CircularBuffer cb_rqc(cb_rq);
    CircularBuffer cb_sinkc(cb_sink);
    cb_writer.reserve_back(MI);
    cb_compute.reserve_back(MI);
    const uint32_t pos_l1 = cb_writer.get_write_ptr(), posc_l1 = cb_compute.get_write_ptr();
    {
    DeviceZoneScopedN("S_R_POS");
    if constexpr (SDPA_BCAST) {
        // (MI == 1) this item's position stick from the broadcast slot
        volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage + item * SLOT);
        volatile tt_l1_ptr uint32_t* d0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_l1);
        volatile tt_l1_ptr uint32_t* d1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(posc_l1);
        for (uint32_t w = 0; w < pos_stick_bytes / 4; ++w) {
            const uint32_t v = src[w];
            d0[w] = v;
            d1[w] = v;
        }
    } else {
        for (uint32_t m = 0; m < MI; ++m) {
            noc.async_read(
                pos_reader,
                CoreLocalMem<uint32_t>(pos_l1 + m * pos_stick_bytes),
                pos_stick_bytes,
                {.page_id = item + m},
                {});
            noc.async_read(
                pos_reader,
                CoreLocalMem<uint32_t>(posc_l1 + m * pos_stick_bytes),
                pos_stick_bytes,
                {.page_id = item + m},
                {});
        }
        noc.async_read_barrier();
    }
    cb_writer.push_back(MI);
    cb_compute.push_back(MI);
    }

    volatile tt_l1_ptr uint32_t* pos_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_l1);
    constexpr uint32_t PW = pos_stick_bytes / 4;
    uint32_t lim[MI * R];
    uint32_t max_pos = 0, min_lim = 0;
    if (!rows_sdpa_limits<MI * R>([&](uint32_t j) { return pos_ptr[(j / R) * PW + j % R]; }, lim, max_pos, min_lim)) {
        continue;  // every row skipped
    }

    uint32_t min_lo = max_pos;
    for (uint32_t j = 0; j < MI * R; ++j) {
        const uint32_t lo = inkling_lo(lim[j], W);
        min_lo = lo < min_lo ? lo : min_lo;
    }
    const uint32_t Sk_chunk_t_dynamic = get_dynamic_Sk_chunk_t<Sk_chunk_t, max_dynamic_chunk_size>(max_pos);
    const uint32_t k_chunk_size_dynamic = Sk_chunk_t_dynamic * tt::constants::TILE_HEIGHT;
    // the window [min_lo, max_pos] covers every row's window (the stock helper's sliding-window workload)
    const uint32_t W_eff = W == 0 ? 0 : max_pos - min_lo + 1;
    uint32_t cores_eff;
    auto [PSt, k_num_chunks, k_chunk_start, k_chunk_end, window_start_unaligned, window_start_chunk] =
        inkling_workload(max_pos, core_num_in_reduce, num_cores_per_head, k_chunk_size_dynamic, W_eff, cores_eff);
    if (k_chunk_start == k_chunk_end) {
        continue;
    }

    // the stock read throttle (a NOC barrier every (512 / readers) x 1152 / tile bytes reads) for the readers active at
    // once: min(chunks, cores per unit) per unit (prefill: one per unit), at least 16 (a short decode step: K / V reads
    // 6.8 -> 1.4 us per chunk); the stock op assumes every program core reads
    uint32_t readers = (k_num_chunks < cores_eff ? k_num_chunks : cores_eff) * n_groups;
    readers = readers < 16 ? 16 : (readers > num_cores ? num_cores : readers);
    uint32_t barrier_threshold = ((512 / readers) * 1152) / q_tile_bytes;
    barrier_threshold = barrier_threshold ? barrier_threshold : 1;
    uint32_t barrier_count = 0;

    // the page-table row up to the blocks the unit attends
    const uint32_t pt_need = (max_pos / (block_size_t * 32) + 1) * 4;
    uint32_t pt_nbytes = (pt_need + 63) & ~63u;
    pt_nbytes = pt_nbytes < page_table_page_size ? pt_nbytes : page_table_page_size;
    page_table_ptr = pt_l1;
    if (SDPA_BCAST && pt_need <= SDPA_BCAST_PT) {
        page_table_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage + item * SLOT + pos_stick_bytes);
    } else if (pt_row != pt_item || pt_nbytes > pt_bytes) {
        noc_async_read(pt_reader.get_noc_addr(pt_row), reinterpret_cast<uint32_t>(pt_l1), pt_nbytes);
        pt_item = pt_row;
        pt_bytes = pt_nbytes;
    }

    {
    DeviceZoneScopedN("S_R_Q");
    // Q / Rq / sink: this kv head's G rows of each row's tiles -> rows r*G+i of a tile (item m's Q at tile row m, Rq
    // at tiles m * R .., sink at tile m); rows past R x G are zero for R > 1 (Rq tile j must hold only row j's heads:
    // the bias sums over j) and left as they are for R == 1 (rows are independent through QK, softmax, PV and
    // the reduction, and never written out)
    cb_q.reserve_back(MI * DHt);
    cb_rqc.reserve_back(MI * RP);
    cb_sinkc.reserve_back(MI);
    uint32_t scr_q, scr_rq, scr_sink;  // the source tiles of item m (scratch or broadcast slot)
    if constexpr (SDPA_BCAST) {
        scr_q = stage + item * SLOT + SLOT_TILES_OFF;
    } else {
        cb_scr.reserve_back(R * (DHt + 2) + 1);
        scr_q = (cb_scr.get_write_ptr() + 63) & ~63u;  // the scratch CB has one spare tile
    }
    scr_rq = scr_q + R * DHt * q_tile_bytes;
    scr_sink = scr_rq + R * q_tile_bytes;
    for (uint32_t m = 0; m < MI; ++m) {
        if (!SDPA_BCAST) {
            for (uint32_t t = 0; t < R * DHt; ++t) {
                gather_rows(q_reader.get_noc_addr((item + m) * R * DHt + t), scr_q + t * q_tile_bytes);
            }
            for (uint32_t t = 0; t < R; ++t) {
                gather_rows(rq_reader.get_noc_addr((item + m) * R + t), scr_rq + t * q_tile_bytes);
                gather_rows(sink_reader.get_noc_addr((item + m) * R + t), scr_sink + t * q_tile_bytes);
            }
        }
        const uint32_t q_l1 = cb_q.get_write_ptr() + m * DHt * q_tile_bytes;
        const uint32_t rq_l1 = cb_rqc.get_write_ptr() + m * RP * q_tile_bytes;
        const uint32_t sink_l1 = cb_sinkc.get_write_ptr() + m * q_tile_bytes;
        if constexpr (R > 1) {
            for (uint32_t d = 0; d < DHt; ++d) {
                fill_tile_zeros<q_tile_bytes, false>(noc, cb_q_in, m * DHt + d);
            }
            for (uint32_t j = 0; j < RP; ++j) {
                fill_tile_zeros<q_tile_bytes, false>(noc, cb_rq, m * RP + j);
            }
            fill_tile_zeros<q_tile_bytes, false>(noc, cb_sink, m);
            noc.write_zeros_l1_barrier();
        }
        noc.async_read_barrier();  // (the gathered rows and the page table)
        for (uint32_t d = 0; d < DHt; ++d) {
            for (uint32_t r = 0; r < R; ++r) {
                place_rows(scr_q + (r * DHt + d) * q_tile_bytes, q_l1 + d * q_tile_bytes, r);
            }
        }
        for (uint32_t j = 0; j < R; ++j) {
            if constexpr (SDPA_PAIR) {
                place_pair(scr_rq + j * q_tile_bytes, rq_l1 + (j / 2) * q_tile_bytes, j);
            } else {
                place_rows(scr_rq + j * q_tile_bytes, rq_l1 + j * q_tile_bytes, j);
            }
            place_rows(scr_sink + j * q_tile_bytes, sink_l1, j);
        }
        noc.async_read_barrier();
    }
    cb_q.push_back(MI * DHt);
    cb_rqc.push_back(MI * RP);
    cb_sinkc.push_back(MI);
    }

    // per chunk: K then V (the compute needs them in that order; the writer reads the bias-table tiles)
    const auto k_reader = TensorAccessor(k_args, k_addr);
    const auto v_reader = TensorAccessor(v_args, v_addr);
    CircularBuffer cb_kc(cb_k_in);
    CircularBuffer cb_vc(cb_v_in);
    auto throttle = [&]() {
        if (++barrier_count == barrier_threshold) {
            noc_async_read_barrier();
            barrier_count = 0;
        }
    };
    DeviceZoneScopedN("S_R_KV");
    for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
        uint32_t k_row0, Skc;  // this chunk's key tiles (SDPA_TRIM: the last one only its live tiles)
        inkling_chunk_tiles(
            k_chunk, Sk_chunk_t_dynamic, max_pos, min_lo, k_chunk + 1 == k_chunk_end, k_row0, Skc, k_num_chunks,
            window_start_chunk, cores_eff);
        const uint32_t k_chunk_tiles = Skc * DHt;
        const uint32_t v_chunk_tiles = Skc * vDHt;
        // K transposed: row t of the chunk -> column t of each of the DHt tile rows (the stock read_k)
        {
            DeviceZoneScopedN("S_R_K");
            cb_kc.reserve_back(k_chunk_tiles);
            barrier_count = 0;
            const uint32_t k_l1 = cb_kc.get_write_ptr();
            for (uint32_t row = 0; row < Skc; ++row) {
                uint32_t tile = virtual_seq_tile_id_to_physical_tile_id<uint32_t, num_kv_heads, block_size_t, DHt, 0>(
                    k_row0 + row, cur_head, page_table_ptr);
                uint32_t dst = k_l1 + row * k_tile_bytes;
                for (uint32_t col = 0; col < DHt; ++col, ++tile, dst += Skc * k_tile_bytes) {
                    noc_async_read(k_reader.get_noc_addr(tile), dst, k_tile_bytes);
                    throttle();
                }
            }
            noc_async_read_barrier();
            cb_kc.push_back(k_chunk_tiles);
        }
        {
            DeviceZoneScopedN("S_R_V");
            cb_vc.reserve_back(v_chunk_tiles);
            barrier_count = 0;
            uint32_t dst = cb_vc.get_write_ptr();
            for (uint32_t row = 0; row < Skc; ++row) {
                uint32_t tile = virtual_seq_tile_id_to_physical_tile_id<uint32_t, num_kv_heads, block_size_t, vDHt, 0>(
                    k_row0 + row, cur_head, page_table_ptr);
                for (uint32_t col = 0; col < vDHt; ++col, ++tile, dst += v_tile_bytes) {
                    noc_async_read(v_reader.get_noc_addr(tile), dst, v_tile_bytes);
                    throttle();
                }
            }
            noc_async_read_barrier();
            cb_vc.push_back(v_chunk_tiles);
        }
    }
    }  // units
}
