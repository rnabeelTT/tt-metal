// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Router tail of one decode row, upper half (BRISC, NOC1; moe_decode/union.py RouteRows): the top-k of logits
// [E / 2, E) of row r, handed to the reader through cb_part (k order keys, k ids).
//
// runtime args: [logits_addr, row, (ROUTER_SIGB) bias_addr]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "router_topk.hpp"

void kernel_main() {
    const uint32_t logits_addr = get_arg_val<uint32_t>(0);
    const uint32_t row = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_lg = get_compile_time_arg_val(0);
    constexpr uint32_t cb_part = get_compile_time_arg_val(1);
    constexpr uint32_t num_experts = get_compile_time_arg_val(2);
    constexpr uint32_t topk = get_compile_time_arg_val(3);
    constexpr auto lg_args = TensorAccessorArgs<4>();

    const auto s_lg = TensorAccessor(lg_args, logits_addr, moe_topk::LG_PAGE);
    const uint32_t roff = moe_topk::lg_row_off(row);  // row r, columns 0..15 (columns 16..31: + LG_FACE1)
    cb_reserve_back(cb_lg, 1);
    const uint32_t lg = get_write_ptr(cb_lg);
    for (uint32_t t = num_experts / 64; t < num_experts / 32; ++t) {
        noc_async_read(s_lg.get_noc_addr(t, roff), lg + t * moe_topk::LG_TILE_L1, moe_topk::LG_HALF);
        noc_async_read(
            s_lg.get_noc_addr(t, roff + moe_topk::LG_FACE1), lg + t * moe_topk::LG_TILE_L1 + moe_topk::LG_HALF,
            moe_topk::LG_HALF);
    }
    const int32_t* bias = nullptr;
#ifdef ROUTER_SIGB
    const uint32_t bias_l1 = lg + (num_experts / 32) * moe_topk::LG_TILE_L1;
    noc_async_read(get_noc_addr_from_bank_id<true>(0, get_arg_val<uint32_t>(2)), bias_l1, 4 * num_experts);
    bias = reinterpret_cast<const int32_t*>(bias_l1);
#endif
    noc_async_read_barrier();
    invalidate_l1_cache();
    cb_reserve_back(cb_part, 1);
    uint32_t* part = reinterpret_cast<uint32_t*>(get_write_ptr(cb_part));
    moe_topk::route_select<topk>(lg, bias, num_experts / 2, num_experts, part, part + topk);
    cb_push_back(cb_part, 1);
}
