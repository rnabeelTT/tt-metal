// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Router tail of one decode row (NCRISC, NOC0) for the multi-row fused MoE (moe_decode/union.py RouteRows): row r of
// the router logits [M, E + 32] (BF16 TILE; column E = the shared-expert gate logit): top-k of logits [0, E / 2) here
// and [E / 2, E) on the writer (cb_part), merged, softmax over the selected, shared gate sigmoid. Writes the row's
// route record into page r of the route table (UINT16 row-major [M, 32]): k ids at [0, k), the shared-expert weight
// at 16, the k routed weights at [17, 17 + k) (BF16 bits).
//
// ROUTER_SIGB (GLM-4.5-Air): FP32 logits [M, E], sigmoid + selection-bias top-k, normalized weights, shared weight 1
// (router_topk.hpp; bias: Q23 int32 [E], one DRAM page in bank 0).
//
// ROUTE_TILES (a port's own expert paths, moe_decode/topk.py RouteScores): the k ids / weights go to row `row` of
// [1, 1, M, k] UINT16 / BF16 TILE outputs instead of the route table (the second tensor accessor = the ids).
//
// runtime args: [logits_addr, table_or_ids_addr, row, (ROUTER_SIGB) bias_addr, (ROUTE_TILES) scores_addr]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "router_topk.hpp"

void kernel_main() {
    const uint32_t logits_addr = get_arg_val<uint32_t>(0);
    const uint32_t table_addr = get_arg_val<uint32_t>(1);
    const uint32_t row = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_scr = get_compile_time_arg_val(0);
    constexpr uint32_t cb_part = get_compile_time_arg_val(1);
    constexpr uint32_t num_experts = get_compile_time_arg_val(2);
    constexpr uint32_t topk = get_compile_time_arg_val(3);
    constexpr auto lg_args = TensorAccessorArgs<4>();
    constexpr auto table_args = TensorAccessorArgs<lg_args.next_compile_time_args_offset()>();
#ifdef ROUTER_SIGB
    constexpr uint32_t a_scores = 4;
#else
    constexpr uint32_t a_scores = 3;
#endif

    const auto s_lg = TensorAccessor(lg_args, logits_addr, moe_topk::LG_PAGE);
    const uint32_t roff = moe_topk::lg_row_off(row);  // row r, columns 0..15 (columns 16..31: + LG_FACE1)
#ifdef ROUTE_TILES
    constexpr auto scores_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    const auto s_ids = TensorAccessor(table_args, table_addr, 2048);
    const auto s_scores = TensorAccessor(scores_args, get_arg_val<uint32_t>(a_scores), 2048);
#else
    const auto s_table = TensorAccessor(table_args, table_addr, 64);
#endif

    // [logits row: lg_tiles x LG_TILE_L1 bytes][route record: 64 bytes][(ROUTER_SIGB) bias: 4 E bytes]
    cb_reserve_back(cb_scr, 1);
    const uint32_t lg = get_write_ptr(cb_scr);
    constexpr uint32_t lg_tiles = moe_topk::lg_tiles(num_experts);
    const uint32_t rec = lg + lg_tiles * moe_topk::LG_TILE_L1;
    for (uint32_t t = 0; t < lg_tiles; ++t) {  // the whole row: the weights read the winners of both halves
        noc_async_read(s_lg.get_noc_addr(t, roff), lg + t * moe_topk::LG_TILE_L1, moe_topk::LG_HALF);
        noc_async_read(
            s_lg.get_noc_addr(t, roff + moe_topk::LG_FACE1), lg + t * moe_topk::LG_TILE_L1 + moe_topk::LG_HALF,
            moe_topk::LG_HALF);
    }
    const int32_t* bias = nullptr;
#ifdef ROUTER_SIGB
    const uint32_t bias_l1 = moe_topk::align64(rec + 64);
    noc_async_read(get_noc_addr_from_bank_id<true>(0, get_arg_val<uint32_t>(3)), bias_l1, 4 * num_experts);
    bias = reinterpret_cast<const int32_t*>(bias_l1);
#endif
    noc_async_read_barrier();
    invalidate_l1_cache();

    uint32_t keys[topk], ids[topk];
    moe_topk::route_select<topk>(lg, bias, 0, num_experts / 2, keys, ids);
    cb_wait_front(cb_part, 1);
    invalidate_l1_cache();
    const uint32_t* part = reinterpret_cast<const uint32_t*>(get_read_ptr(cb_part));
    moe_topk::merge<topk>(keys, ids, part, part + topk);
    cb_pop_front(cb_part, 1);
    uint16_t w[topk];
    const uint16_t w_shared = moe_topk::route_weights<topk, num_experts>(lg, ids, w);

    volatile tt_l1_ptr uint16_t* r = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rec);
    for (uint32_t j = 0; j < 32; ++j) {
        r[j] = 0;
    }
    for (uint32_t j = 0; j < topk; ++j) {
        r[j] = static_cast<uint16_t>(ids[j]);
        r[17 + j] = w[j];
    }
    r[16] = w_shared;
#ifdef ROUTE_TILES
    // row r of a 2-byte [32, 32] tile: face 2 (r / 16), row r % 16; the ids at rec[0, k), the weights at rec[17, 17 + k)
    for (uint32_t j = 0; j < topk; ++j) {
        r[32 + j] = r[17 + j];  // 64-byte aligned copy of the weights (rec + 64)
    }
    const uint32_t off = (row / 16) * 1024 + (row % 16) * 32;
    noc_async_write(rec, s_ids.get_noc_addr(0, off), 2 * topk);
    noc_async_write(rec + 64, s_scores.get_noc_addr(0, off), 2 * topk);
#else
    noc_async_write(rec, s_table.get_noc_addr(row), 64);
#endif
    noc_async_write_barrier();
}
