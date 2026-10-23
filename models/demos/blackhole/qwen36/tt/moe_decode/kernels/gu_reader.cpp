// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Expert gate|up weight streamer (NCRISC, NOC0) of the fused MoE decode (moe_decode/experts.py ExpertGateUp).
//
// 1. Reads row 0 of the router logits [32, E + 32] (BF16 TILE; column E = the shared-expert gate logit).
// 2. If this core owns the shared expert (slot 0), issues its two weight columns first: they stream while the
//    router tail runs.
// 3. Router tail, split with the writer (BRISC): this core takes the top-k of logits [0, E / 2), the writer that of
//    [E / 2, E) (cb_part); merged, softmax over the selected, shared gate sigmoid. Hands [k ids, k + 1 BF16 weights]
//    to the writer through cb_route (slot 0 = the shared expert, slots 1..k = the routed experts in top-k order).
// 4. Streams this core's remaining columns straight from the ttnn expert tensors (no decode copy): a [1, E, K, N]
//    TILE tensor interleaved over the DRAM banks keeps page p in bank p % banks, so when N tiles is a multiple of the
//    bank count, output column n of expert e is a run of K tiles in bank n % banks (K * N / banks tiles per expert and
//    bank, row-major over (k, local column)). This core owns column pair j: up column j and gate column j + N / 2 of
//    each of its slots (runtime list). Each column is one cb_w block of kt tiles, read in page_bytes packets,
//    transaction-id pipelined (num_buffers - 2 columns in flight ahead of compute).
//
// runtime args: [logits_addr, w_addr, ws_addr, pair j, vc, num_slots, slot...]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "noc_state.hpp"
#include "router_topk.hpp"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t logits_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);   // routed [1, E, K, N]
    const uint32_t ws_addr = get_arg_val<uint32_t>(2);  // shared [1, 1, K, N]
    const uint32_t pair = get_arg_val<uint32_t>(3);
    const uint32_t vc = get_arg_val<uint32_t>(4);
    const uint32_t num_slots = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_route = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(2);
    constexpr uint32_t kt = get_compile_time_arg_val(3);       // K tiles per column
    constexpr uint32_t n_tiles = get_compile_time_arg_val(4);  // N tiles of the packed [up | gate] weight
    constexpr uint32_t num_experts = get_compile_time_arg_val(5);
    constexpr uint32_t topk = get_compile_time_arg_val(6);
    constexpr uint32_t cb_part = get_compile_time_arg_val(7);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t banks = get_compile_time_arg_val(10);
    constexpr uint32_t num_buffers = get_compile_time_arg_val(11);  // cb_w columns
    constexpr auto logits_args = TensorAccessorArgs<12>();

    static_assert(n_tiles % banks == 0, "the column-in-one-bank layout needs N tiles to be a multiple of the banks");
    constexpr uint32_t m = n_tiles / banks;  // local columns per bank and expert
    static_assert(m == 1 || m == 2, "up|gate column pairs: in two banks (m = 1) or one bank (m = 2)");
    // a block: m == 1: one column (kt tiles; up and gate in banks j and j + banks / 2); m == 2: the pair's [kt, 2]
    // run in bank j (up = local column 0, gate = local column 1, row-major), read whole
    constexpr uint32_t blk_tiles = kt * m;
    constexpr uint32_t col_bytes = blk_tiles * tile_bytes;
    constexpr uint32_t pages_per_col = col_bytes / page_bytes;
    static_assert(pages_per_col * page_bytes == col_bytes, "page size must divide the block");
    constexpr uint32_t lg_tiles = num_experts / 32 + 1;

    // ---- logits row ----
    cb_reserve_back(cb_scr, 1);
    const uint32_t lg = get_write_ptr(cb_scr);
    const auto s_lg = TensorAccessor(logits_args, logits_addr, 2048);
    for (uint32_t t = 0; t < lg_tiles; ++t) {
        noc_async_read(s_lg.get_noc_addr(t, 0), lg + t * 64, 32);
        noc_async_read(s_lg.get_noc_addr(t, 512), lg + t * 64 + 32, 32);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();

    // ---- column stream: per slot, up column j then gate column j + n_tiles / 2 ----
    constexpr uint32_t extra_in_flight = num_buffers - 2;
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    cb_reserve_back(cb_w, blk_tiles * (extra_in_flight + 1));
    uint32_t l1_write = get_write_ptr(cb_w);
    const uint32_t cb_base = l1_write;
    const uint32_t cb_end = cb_base + num_buffers * col_bytes;
    uint32_t free_blocks = num_buffers;
    uint32_t trid = 1;
    uint32_t trid_to_wait = 1;
    auto stream_block = [&](uint32_t base, uint32_t e, uint32_t n) {
        // page (e, k, n) = (e * kt + k) * n_tiles + n -> bank n % banks, bank-local page (e * kt + k) * m + n / banks
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
    };
    auto stream_slot = [&](uint32_t base, uint32_t e) {
        stream_block(base, e, pair);
        if constexpr (m == 1) {
            stream_block(base, e, pair + n_tiles / 2);
        }
    };

    uint32_t first = 0;
    if (num_slots > 0 && get_arg_val<uint32_t>(6) == 0) {  // the shared expert streams during the router tail
        stream_slot(ws_addr, 0);
        first = 1;
    }

    uint32_t keys[topk], ids[topk];
    {
        DeviceZoneScopedN("tail");
    // ---- router tail: top-k of [0, E / 2) here, of [E / 2, E) on the writer, merged ----
    const uint32_t* words = reinterpret_cast<const uint32_t*>(lg);
    moe_topk::select_range<topk>(words, 0, num_experts / 4, keys, ids);
    cb_wait_front(cb_part, 1);
    invalidate_l1_cache();
    const uint32_t* part = reinterpret_cast<const uint32_t*>(get_read_ptr(cb_part));
    moe_topk::merge<topk>(keys, ids, part, part + topk);
    cb_pop_front(cb_part, 1);
    uint16_t w[topk];
    moe_topk::softmax_selected<topk>(words, ids, w);
    cb_reserve_back(cb_route, 1);
    volatile tt_l1_ptr uint16_t* route = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_route));
    for (uint32_t j = 0; j < topk; ++j) {
        route[j] = static_cast<uint16_t>(ids[j]);
        route[16 + 1 + j] = w[j];
    }
    route[16] = moe_topk::sigmoid_bf16(reinterpret_cast<const uint16_t*>(lg)[num_experts]);
    cb_push_back(cb_route, 1);
    }

    for (uint32_t i = first; i < num_slots; ++i) {
        const uint32_t e = ids[get_arg_val<uint32_t>(6 + i) - 1];
        stream_slot(w_addr, e);
    }
    if (num_slots > 0) {
        for (uint32_t i = 0; i < num_buffers - free_blocks; ++i) {  // the columns still in flight
            noc_async_read_barrier_with_trid(trid_to_wait);
            cb_push_back(cb_w, blk_tiles);
            trid_to_wait = trid_to_wait == num_buffers ? 1 : trid_to_wait + 1;
        }
    }
    restore_read_cmd_buf();
}
