// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// One-row gate|up weight streamer, one-bank layout (NCRISC; moe_decode/experts.py ExpertGateUp1B, m odd: a pair's up
// column in bank b, its gate column in bank b +- banks / 2). The route record is in the route buffer already (an earlier
// program took the top-k: ROUTE_IN only); this core streams, per slot it owns, its ONE bank's [kt, m] block of the
// slot's expert (gu_layout.hpp: bank-local columns b, b + banks, .. of the expert's [K, N] run) into cb_w, transaction-id
// pipelined as gu_reader.cpp (num_buffers - 2 blocks in flight ahead of compute). Pushes the record to the writer
// (cb_route).
//
// runtime args: [route_addr, w_addr, bank, vc, num_slots, slot...]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "router_topk.hpp"

void kernel_main() {
    const uint32_t route_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t bank = get_arg_val<uint32_t>(2);
    const uint32_t vc = get_arg_val<uint32_t>(3);
    const uint32_t num_slots = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_route = get_compile_time_arg_val(1);
    constexpr uint32_t kt = get_compile_time_arg_val(2);
    constexpr uint32_t m = get_compile_time_arg_val(3);  // columns per bank and expert
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t num_buffers = get_compile_time_arg_val(6);
    constexpr uint32_t topk = get_compile_time_arg_val(7);
    constexpr auto route_args = TensorAccessorArgs<8>();

    constexpr uint32_t blk_tiles = kt * m;
    constexpr uint32_t blk_bytes = blk_tiles * tile_bytes;
    constexpr uint32_t pages = blk_bytes / page_bytes;
    static_assert(pages * page_bytes == blk_bytes, "page size must divide the block");

    cb_reserve_back(cb_route, 1);
    const uint32_t rl1 = get_write_ptr(cb_route);
    const auto s_rt = TensorAccessor(route_args, route_addr, moe_topk::REC_BYTES);
    noc_async_read(s_rt.get_noc_addr(0), rl1, moe_topk::REC_BYTES);
    noc_async_read_barrier();
    invalidate_l1_cache();
    uint32_t ids[topk];
    {
        volatile tt_l1_ptr uint16_t* route = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rl1);
        for (uint32_t j = 0; j < topk; ++j) {
            ids[j] = route[j];
        }
    }
    cb_push_back(cb_route, 1);

    constexpr uint32_t extra_in_flight = num_buffers - 2;
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    cb_reserve_back(cb_w, blk_tiles * (extra_in_flight + 1));
    uint32_t l1_write = get_write_ptr(cb_w);
    const uint32_t cb_base = l1_write;
    const uint32_t cb_end = cb_base + num_buffers * blk_bytes;
    uint32_t free_blocks = num_buffers;
    uint32_t trid = 1;
    uint32_t trid_to_wait = 1;
    const uint64_t bank_noc = get_noc_addr_from_bank_id<true>(bank, w_addr);
    for (uint32_t i = 0; i < num_slots; ++i) {
        const uint32_t e = ids[get_arg_val<uint32_t>(5 + i) - 1];
        uint32_t src = e * blk_bytes;  // expert e's [kt, m] run in this bank
        noc_async_read_one_packet_set_state<true>(bank_noc, page_bytes, vc);
        noc_async_read_set_trid(trid);
        for (uint32_t p = 0; p < pages; ++p) {
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
    if (num_slots > 0) {
        for (uint32_t i = 0; i < num_buffers - free_blocks; ++i) {  // the blocks still in flight
            noc_async_read_barrier_with_trid(trid_to_wait);
            cb_push_back(cb_w, blk_tiles);
            trid_to_wait = trid_to_wait == num_buffers ? 1 : trid_to_wait + 1;
        }
    }
    noc_async_read_one_packet_set_state<true>(get_noc_addr(MEM_ZEROS_BASE), 0, 1);
    noc_async_read_set_trid(0);
}
