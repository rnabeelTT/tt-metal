// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Multi-row expert down-projection weight streamer (NCRISC, NOC0) of the fused MoE decode (moe_decode/union.py
// ExpertDownRows). As dn_reader.cpp, over the union slots of M route records (slot 0 = the shared expert, 1 + i =
// union expert i, see router_topk.hpp route_union): per slot and K row one read of `cols` tiles, slots stacked along
// K in chunks of `chunk` slots (one cb_w push per chunk; the compute accumulates across chunks).
//
// NO_SHARED: routed experts only (union slots 1 .. U).
//
// EP_EXPERTS: only this die's window of the union (local ids).
//
// runtime args: [w_addr, ws_addr, table_addr, bank, c0, vc, (EP_EXPERTS) ep_offset_addr]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "noc_state.hpp"
#include "router_topk.hpp"

void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t ws_addr = get_arg_val<uint32_t>(1);
    const uint32_t table_addr = get_arg_val<uint32_t>(2);
    const uint32_t bank = get_arg_val<uint32_t>(3);
    const uint32_t c0 = get_arg_val<uint32_t>(4);
    const uint32_t vc = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(1);
    constexpr uint32_t kt = get_compile_time_arg_val(2);
    constexpr uint32_t m = get_compile_time_arg_val(3);
    constexpr uint32_t cols = get_compile_time_arg_val(4);
    constexpr uint32_t topk = get_compile_time_arg_val(5);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t rows = get_compile_time_arg_val(7);
    constexpr uint32_t chunk = get_compile_time_arg_val(8);
    constexpr auto table_args = TensorAccessorArgs<9>();

    constexpr uint32_t row_bytes = cols * tile_bytes;
    constexpr uint32_t slot_tiles = kt * cols;

    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    const auto s_table = TensorAccessor(table_args, table_addr, 64);
    for (uint32_t r = 0; r < rows; ++r) {
        noc_async_read(s_table.get_noc_addr(r), scr + 64 * r, 64);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    uint16_t uni[rows * topk];
#ifdef EP_EXPERTS
    const uint32_t U = moe_topk::window_union(
        uni, moe_topk::route_union<topk>(reinterpret_cast<const uint16_t*>(scr), rows, uni),
        moe_topk::read_ep_offset(get_arg_val<uint32_t>(6), scr + 64 * rows), EP_EXPERTS);
#else
    const uint32_t U = moe_topk::route_union<topk>(reinterpret_cast<const uint16_t*>(scr), rows, uni);
#endif
#ifdef NO_SHARED
    constexpr uint32_t S0 = 1;  // no shared expert: slots 1 .. (routed only)
#else
    constexpr uint32_t S0 = 0;
#endif
    const uint32_t slots = U + 1 - S0;

    const uint64_t routed_noc = get_noc_addr_from_bank_id<true>(bank, w_addr);
    const uint64_t shared_noc = get_noc_addr_from_bank_id<true>(bank, ws_addr);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    // chunks alternate between transaction ids 1 and 2: the next chunk is in flight while compute takes this one
    // cb_w holds exactly two full chunks: a chunk starts at either half
    const uint32_t cb_base = get_write_ptr(cb_w);
    const uint32_t cb_end = cb_base + 2 * chunk * slot_tiles * tile_bytes;
    uint32_t prev_tiles = 0;
    uint32_t trid = 1;
    for (uint32_t s0 = 0; s0 < slots; s0 += chunk) {
        const uint32_t cs = slots - s0 < chunk ? slots - s0 : chunk;
        cb_reserve_back(cb_w, cs * slot_tiles + prev_tiles);
        uint32_t l1 = get_write_ptr(cb_w) + prev_tiles * tile_bytes;
        if (l1 >= cb_end) {
            l1 -= cb_end - cb_base;
        }
        noc_async_read_set_trid(trid);
        for (uint32_t i = s0; i < s0 + cs; ++i) {
            const uint32_t s = S0 + i;  // union slot
            const uint64_t bank_noc = s == 0 ? shared_noc : routed_noc;
            const uint32_t e = s == 0 ? 0 : uni[s - 1];
            noc_async_read_one_packet_set_state<true>(bank_noc, row_bytes, vc);
            uint32_t src = (e * kt * m + c0) * tile_bytes;
            for (uint32_t k = 0; k < kt; ++k) {
                noc_async_read_one_packet_with_state_with_trid((uint32_t)bank_noc, src, l1, trid);
                src += m * tile_bytes;
                l1 += row_bytes;
            }
        }
        if (prev_tiles) {
            noc_async_read_barrier_with_trid(trid == 1 ? 2 : 1);
            cb_push_back(cb_w, prev_tiles);
        }
        prev_tiles = cs * slot_tiles;
        trid = trid == 1 ? 2 : 1;
    }
    noc_async_read_barrier_with_trid(trid == 1 ? 2 : 1);
    cb_push_back(cb_w, prev_tiles);
    restore_read_cmd_buf();
}
