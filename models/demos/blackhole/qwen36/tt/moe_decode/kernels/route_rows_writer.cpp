// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Router tail of one decode row, upper half (BRISC, NOC1; moe_decode/union.py RouteRows): the top-k of logits
// [E / 2, E) of row r, handed to the reader through cb_part (k order keys, k ids).
//
// runtime args: [logits_addr, row]

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

    const auto s_lg = TensorAccessor(lg_args, logits_addr, 2048);
    const uint32_t roff = (row / 16) * 1024 + (row % 16) * 32;  // row r, columns 0..15 (columns 16..31: + 512)
    cb_reserve_back(cb_lg, 1);
    const uint32_t lg = get_write_ptr(cb_lg);
    for (uint32_t t = num_experts / 64; t < num_experts / 32; ++t) {
        noc_async_read(s_lg.get_noc_addr(t, roff), lg + t * 64, 32);
        noc_async_read(s_lg.get_noc_addr(t, roff + 512), lg + t * 64 + 32, 32);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    cb_reserve_back(cb_part, 1);
    uint32_t* part = reinterpret_cast<uint32_t*>(get_write_ptr(cb_part));
    moe_topk::select_range<topk>(reinterpret_cast<const uint32_t*>(lg), num_experts / 4, num_experts / 2, part, part + topk);
    cb_push_back(cb_part, 1);
}
