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
    constexpr auto q_args = TensorAccessorArgs<14>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto page_table_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    constexpr auto rq_args = TensorAccessorArgs<page_table_args.next_compile_time_args_offset()>();
    constexpr auto tbl_args = TensorAccessorArgs<rq_args.next_compile_time_args_offset()>();
    static_assert(R * G <= 32, "the rows' query heads of one kv head share one tile");

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;
    constexpr uint32_t cb_writer_cur_pos = tt::CBIndex::c_8;
    constexpr uint32_t cb_id_page_table = tt::CBIndex::c_9;
    constexpr uint32_t cb_q_scratch = tt::CBIndex::c_10;
    constexpr uint32_t cb_compute_cur_pos = tt::CBIndex::c_15;
    constexpr uint32_t cb_rq = tt::CBIndex::c_4;    // R tiles: row j's G query heads' r (16 cols) at tile rows j*G..
    constexpr uint32_t cb_tbl = tt::CBIndex::c_11;  // per band chunk: R x Sk bias-table tiles (row-major [j][t])

    uint32_t arg_idx = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t k_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t v_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t pos_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t page_table_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t page_table_page_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cur_head = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_num_in_reduce = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t rq_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t tbl_addr = get_arg_val<uint32_t>(arg_idx++);

    // cur_pos stick -> the writer's CB and (L1 -> L1) the compute's CB (the stock op's #44366 split)
    CircularBuffer cb_writer(cb_writer_cur_pos);
    cb_writer.reserve_back(1);
    const uint32_t pos_l1 = cb_writer.get_write_ptr();
    {
        const auto pos_reader = TensorAccessor(pos_args, pos_addr);
        noc.async_read(pos_reader, CoreLocalMem<uint32_t>(pos_l1), pos_stick_bytes, {.page_id = 0}, {});
        noc.async_read_barrier();
    }
    const uint8_t noc_id = noc.get_noc_id();
    const uint32_t my_noc_x = my_x[noc_id];
    const uint32_t my_noc_y = my_y[noc_id];
    UnicastEndpoint self;
    CircularBuffer cb_compute(cb_compute_cur_pos);
    cb_compute.reserve_back(1);
    noc.async_read(
        self,
        CoreLocalMem<uint32_t>(cb_compute.get_write_ptr()),
        pos_stick_bytes,
        {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = pos_l1},
        {});
    noc.async_read_barrier();
    cb_writer.push_back(1);
    cb_compute.push_back(1);

    volatile tt_l1_ptr uint32_t* pos_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_l1);
    uint32_t lim[R];
    uint32_t max_pos = 0, min_lim = 0;
    if (!rows_sdpa_limits<R>([&](uint32_t j) { return pos_ptr[j]; }, lim, max_pos, min_lim)) {
        return;  // every row skipped
    }

    uint32_t min_lo = max_pos;
    for (uint32_t j = 0; j < R; ++j) {
        const uint32_t lo = inkling_lo(lim[j], W);
        min_lo = lo < min_lo ? lo : min_lo;
    }
    const uint32_t Sk_chunk_t_dynamic = get_dynamic_Sk_chunk_t<Sk_chunk_t, max_dynamic_chunk_size>(max_pos);
    const uint32_t k_chunk_size_dynamic = Sk_chunk_t_dynamic * tt::constants::TILE_HEIGHT;
    // the window [min_lo, max_pos] covers every row's window (the stock helper's sliding-window workload)
    const uint32_t W_eff = W == 0 ? 0 : max_pos - min_lo + 1;
    auto [PSt, k_num_chunks, k_chunk_start, k_chunk_end, window_start_unaligned, window_start_chunk] =
        get_workload_for_core(max_pos, 0, core_num_in_reduce, num_cores_per_head, k_chunk_size_dynamic, W_eff);
    if (k_chunk_start == k_chunk_end) {
        return;
    }

    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);
    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<q_tile_bytes, num_cores>();
    uint32_t barrier_count = 0;

    // Q: this kv head's G head rows of every row's tile row, read from DRAM into a scratch tile at the same in-tile
    // offsets (whole 64-byte row pairs, so source and destination agree mod 64), then rows g*G+i of row r's tiles to
    // rows r*G+i of a zeroed tile (two 32-byte face rows per tile row, L1 -> L1)
    CircularBuffer cb_q(cb_q_in);
    CircularBuffer cb_scr(cb_q_scratch);
    cb_q.reserve_back(DHt);
    cb_scr.reserve_back(R * DHt + 1);
    const uint32_t q_l1 = cb_q.get_write_ptr();
    const uint32_t scr_l1 = (cb_scr.get_write_ptr() + 63) & ~63u;  // the scratch CB has one spare tile
    {
        const auto q_reader = TensorAccessor(q_args, q_addr, q_tile_bytes);
        const uint32_t r0 = cur_head * G, r1 = cur_head * G + G;  // tile rows [r0, r1)
        for (uint32_t t = 0; t < R * DHt; ++t) {
            const uint64_t src = q_reader.get_noc_addr(t);
            for (uint32_t top = 0; top < 32; top += 16) {  // face rows 0-15 (faces 0, 1), 16-31 (faces 2, 3)
                const uint32_t a = r0 > top ? r0 : top;
                const uint32_t b = r1 < top + 16 ? r1 : top + 16;
                if (a >= b) {
                    continue;
                }
                const uint32_t a2 = (a - top) & ~1u, b2 = ((b - top) + 1) & ~1u;  // row pairs within the face
                for (uint32_t half = 0; half < 2; ++half) {
                    const uint32_t off = (top >> 4) * 1024 + half * 512 + a2 * 32;
                    noc_async_read(src + off, scr_l1 + t * q_tile_bytes + off, (b2 - a2) * 32);
                }
            }
        }
    }
    for (uint32_t d = 0; d < DHt; ++d) {
        fill_tile_zeros<q_tile_bytes, false>(noc, cb_q_in, d);
    }
    noc.write_zeros_l1_barrier();
    noc.async_read_barrier();
    for (uint32_t d = 0; d < DHt; ++d) {
        for (uint32_t r = 0; r < R; ++r) {
            const uint32_t src_tile = scr_l1 + (r * DHt + d) * q_tile_bytes, dst_tile = q_l1 + d * q_tile_bytes;
            rows_sdpa_row_runs(cur_head * G, r * G, G, [&](uint32_t s, uint32_t t, uint32_t k) {
                for (uint32_t half = 0; half < 2; ++half) {
                    noc.async_read(
                        self,
                        CoreLocalMem<uint32_t>(dst_tile + rows_sdpa_row_off(t) + half * 512),
                        k * 32,
                        {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = src_tile + rows_sdpa_row_off(s) + half * 512},
                        {});
                }
            });
        }
    }
    noc.async_read_barrier();
    cb_q.push_back(DHt);

    // Rq: tile j = row j's G query heads of this kv head (tile rows j*G .. j*G+G-1, the r vector in columns 0..15),
    // every other row zero, so sum_j Rq_j @ T[phi_j][m_j] gives each row its own bias (tt/relbias.py)
    CircularBuffer cb_rqc(cb_rq);
    cb_rqc.reserve_back(R);
    const uint32_t rq_l1 = cb_rqc.get_write_ptr();
    {
        const auto rq_reader = TensorAccessor(rq_args, rq_addr, q_tile_bytes);
        const uint32_t r0 = cur_head * G, r1 = cur_head * G + G;
        for (uint32_t t = 0; t < R; ++t) {
            const uint64_t src = rq_reader.get_noc_addr(t);
            for (uint32_t top = 0; top < 32; top += 16) {
                const uint32_t a = r0 > top ? r0 : top;
                const uint32_t b = r1 < top + 16 ? r1 : top + 16;
                if (a >= b) {
                    continue;
                }
                const uint32_t a2 = (a - top) & ~1u, b2 = ((b - top) + 1) & ~1u;
                for (uint32_t half = 0; half < 2; ++half) {
                    const uint32_t off = (top >> 4) * 1024 + half * 512 + a2 * 32;
                    noc_async_read(src + off, scr_l1 + t * q_tile_bytes + off, (b2 - a2) * 32);
                }
            }
        }
    }
    for (uint32_t j = 0; j < R; ++j) {
        fill_tile_zeros<q_tile_bytes, false>(noc, cb_rq, j);
    }
    noc.write_zeros_l1_barrier();
    noc.async_read_barrier();
    for (uint32_t j = 0; j < R; ++j) {
        const uint32_t src_tile = scr_l1 + j * q_tile_bytes, dst_tile = rq_l1 + j * q_tile_bytes;
        rows_sdpa_row_runs(cur_head * G, j * G, G, [&](uint32_t s, uint32_t t, uint32_t k) {
            for (uint32_t half = 0; half < 2; ++half) {
                noc.async_read(
                    self,
                    CoreLocalMem<uint32_t>(dst_tile + rows_sdpa_row_off(t) + half * 512),
                    k * 32,
                    {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = src_tile + rows_sdpa_row_off(s) + half * 512},
                    {});
            }
        });
    }
    noc.async_read_barrier();
    cb_rqc.push_back(R);

    // the sequence's page-table row (every verify row shares it: row 0 is read)
    CircularBuffer cb_page_table(cb_id_page_table);
    cb_page_table.reserve_back(1);
    volatile tt_l1_ptr uint32_t* page_table_ptr =
        read_page_table_for_batch(noc, cb_id_page_table, 0, page_table_args, page_table_addr, page_table_page_size);
    cb_page_table.push_back(1);

    const auto k_reader = TensorAccessor(k_args, k_addr);
    const auto v_reader = TensorAccessor(v_args, v_addr);
    const uint32_t k_chunk_tiles = Sk_chunk_t_dynamic * DHt;
    const uint32_t v_chunk_tiles = Sk_chunk_t_dynamic * vDHt;
    const auto tbl_reader = TensorAccessor(tbl_args, tbl_addr, q_tile_bytes);
    CircularBuffer cb_tblc(cb_tbl);
    for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
        const uint32_t k_chunk_start_row_num = k_chunk * Sk_chunk_t_dynamic;
        if (inkling_chunk_band(k_chunk, k_chunk_size_dynamic, min_lim, E)) {
            // R x Sk table tiles: row j, key tile b = k_chunk * Sk + t -> T[phi_j][m = a_j - b] (zero outside [0, M))
            const uint32_t n = R * Sk_chunk_t_dynamic;
            cb_tblc.reserve_back(n);
            uint32_t idx = 0;
            for (uint32_t j = 0; j < R; ++j) {
                const uint32_t a = lim[j] >> 5, phi = lim[j] & 31;
                for (uint32_t t = 0; t < Sk_chunk_t_dynamic; ++t, ++idx) {
                    const uint32_t b = k_chunk_start_row_num + t;
                    if (b <= a && a - b < M_tiles) {
                        noc_async_read(
                            tbl_reader.get_noc_addr(phi * M_tiles + (a - b)),
                            cb_tblc.get_write_ptr() + idx * q_tile_bytes,
                            q_tile_bytes);
                    } else {
                        fill_tile_zeros<q_tile_bytes, false>(noc, cb_tbl, idx);
                    }
                }
            }
            noc.write_zeros_l1_barrier();
            noc.async_read_barrier();
            cb_tblc.push_back(n);
        }
        read_k<cb_k_in, DHt, num_kv_heads, block_size_t, k_tile_bytes, barrier_threshold, false, false, 0>(
            k_chunk_tiles,
            cur_head,
            Sk_chunk_t_dynamic,
            k_chunk_start_row_num,
            k_reader,
            nullptr,
            page_table_ptr,
            barrier_count);
        read_v<cb_v_in, vDHt, num_kv_heads, block_size_t, v_tile_bytes, barrier_threshold, false, false, 0>(
            v_chunk_tiles,
            cur_head,
            Sk_chunk_t_dynamic,
            k_chunk_start_row_num,
            v_reader,
            nullptr,
            page_table_ptr,
            barrier_count);
    }
}
