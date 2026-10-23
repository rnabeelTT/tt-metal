// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Multi-row expert gate|up weight streamer (NCRISC, NOC0) of the fused MoE decode (moe_decode/union.py
// ExpertGateUpRows): M decode rows read every union expert's weights once.
//
// 1. Reads the M route records (route table, RouteRows) and builds the union of the routed experts (ascending; union
//    slot 0 = the shared expert, slot 1 + i = union expert i).
// 2. Reads the identity tile twice into cb_id (the compute loads the per-row weight tiles into DST through it).
// 3. Streams, for each union slot s = slot0, slot0 + step, ... this core owns, up column j and gate column
//    j + N / 2 of the slot's expert, in place from the interleaved ttnn tensor (see gu_reader.cpp), one cb_w block of
//    kt tiles per column (m == 1) or one [kt, 2] block per slot (m == 2, both columns in one bank),
//    transaction-id pipelined.
//
// runtime args: [table_addr, w_addr, ws_addr, id_addr, pair j, slot0, vc]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "noc_state.hpp"
#include "router_topk.hpp"

void kernel_main() {
    const uint32_t table_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t ws_addr = get_arg_val<uint32_t>(2);
    const uint32_t id_addr = get_arg_val<uint32_t>(3);
    const uint32_t pair = get_arg_val<uint32_t>(4);
    const uint32_t slot0 = get_arg_val<uint32_t>(5);
    const uint32_t vc = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t kt = get_compile_time_arg_val(3);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t topk = get_compile_time_arg_val(5);
    constexpr uint32_t rows = get_compile_time_arg_val(6);
    constexpr uint32_t step = get_compile_time_arg_val(7);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t banks = get_compile_time_arg_val(10);
    constexpr uint32_t num_buffers = get_compile_time_arg_val(11);
    constexpr auto table_args = TensorAccessorArgs<12>();
    constexpr auto id_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();

    constexpr uint32_t m = n_tiles / banks;
    static_assert(m == 1 || m == 2, "up|gate column pairs: in two banks (m = 1) or one bank (m = 2)");
    constexpr uint32_t blk_tiles = kt * m;  // a column (m == 1) or the pair's [kt, 2] run in one bank (m == 2)
    constexpr uint32_t col_bytes = blk_tiles * tile_bytes;
    constexpr uint32_t pages_per_col = col_bytes / page_bytes;
    static_assert(pages_per_col * page_bytes == col_bytes, "page size must divide the block");

    // identity tile x 2 (custom_mm takes an even K)
    const auto s_id = TensorAccessor(id_args, id_addr, tile_bytes);
    cb_reserve_back(cb_id, 2);
    noc_async_read(s_id.get_noc_addr(0), get_write_ptr(cb_id), tile_bytes);
    noc_async_read(s_id.get_noc_addr(0), get_write_ptr(cb_id) + tile_bytes, tile_bytes);

    // route table -> union
    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    const auto s_table = TensorAccessor(table_args, table_addr, 64);
    for (uint32_t r = 0; r < rows; ++r) {
        noc_async_read(s_table.get_noc_addr(r), scr + 64 * r, 64);
    }
    noc_async_read_barrier();
    cb_push_back(cb_id, 2);
    invalidate_l1_cache();
    uint16_t uni[rows * topk];
    const uint32_t U = moe_topk::route_union<topk>(reinterpret_cast<const uint16_t*>(scr), rows, uni);

    constexpr uint32_t extra_in_flight = num_buffers - 2;
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    cb_reserve_back(cb_w, blk_tiles * (extra_in_flight + 1));
    uint32_t l1_write = get_write_ptr(cb_w);
    const uint32_t cb_base = l1_write;
    const uint32_t cb_end = cb_base + num_buffers * col_bytes;
    uint32_t free_blocks = num_buffers;
    uint32_t trid = 1;
    uint32_t trid_to_wait = 1;
    for (uint32_t s = slot0; s < U + 1; s += step) {
        const uint32_t base = s == 0 ? ws_addr : w_addr;
        const uint32_t e = s == 0 ? 0 : uni[s - 1];
        for (uint32_t h = 0; h < 3 - m; ++h) {
            const uint32_t n = pair + h * (n_tiles / 2);
            uint32_t src = (e * kt * m + n / banks) * tile_bytes;
            const uint64_t bank_noc = get_noc_addr_from_bank_id<true>(n % banks, base);
            noc_async_read_one_packet_set_state<true>(bank_noc, page_bytes, vc);
            noc_async_read_set_trid(trid);
            for (uint32_t p = 0; p < pages_per_col; ++p) {
                noc_async_read_one_packet_with_state_with_trid((uint32_t)bank_noc, src, l1_write, trid);
                src += page_bytes;
                l1_write += page_bytes;
            }
            if (free_blocks == num_buffers - extra_in_flight) {
                noc_async_read_barrier_with_trid(trid_to_wait);
                cb_push_back(cb_w, blk_tiles);
                trid_to_wait = trid_to_wait == num_buffers ? 1 : trid_to_wait + 1;
                cb_reserve_back(cb_w, blk_tiles * (extra_in_flight + 1));
            } else {
                free_blocks -= 1;
            }
            trid = trid == num_buffers ? 1 : trid + 1;
            if (l1_write >= cb_end) {
                l1_write = cb_base;
            }
        }
    }
    // the columns still in flight: min(streamed, extra_in_flight)
    const uint32_t pending = num_buffers - free_blocks;
    for (uint32_t i = 0; i < pending; ++i) {
        noc_async_read_barrier_with_trid(trid_to_wait);
        cb_push_back(cb_w, blk_tiles);
        trid_to_wait = trid_to_wait == num_buffers ? 1 : trid_to_wait + 1;
    }
    restore_read_cmd_buf();
}
