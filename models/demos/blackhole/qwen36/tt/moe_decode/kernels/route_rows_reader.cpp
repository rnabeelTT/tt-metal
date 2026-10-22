// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Router tail of one decode row (NCRISC, NOC0) for the multi-row fused MoE (moe_decode/union.py RouteRows): row r of
// the router logits [M, E + 32] (BF16 TILE; column E = the shared-expert gate logit): top-k of logits [0, E / 2) here
// and [E / 2, E) on the writer (cb_part), merged, softmax over the selected, shared gate sigmoid. Writes the row's
// route record into page r of the route table (UINT16 row-major [M, 32]): k ids at [0, k), the shared-expert weight
// at 16, the k routed weights at [17, 17 + k) (BF16 bits).
//
// runtime args: [logits_addr, table_addr, row]

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

    const auto s_lg = TensorAccessor(lg_args, logits_addr, 2048);
    const uint32_t roff = (row / 16) * 1024 + (row % 16) * 32;  // row r, columns 0..15 (columns 16..31: + 512)
    const auto s_table = TensorAccessor(table_args, table_addr, 64);

    // [logits row: (E / 32 + 1) x 64 bytes][route record: 64 bytes]
    cb_reserve_back(cb_scr, 1);
    const uint32_t lg = get_write_ptr(cb_scr);
    constexpr uint32_t lg_tiles = num_experts / 32 + 1;
    const uint32_t rec = lg + lg_tiles * 64;
    for (uint32_t t = 0; t < num_experts / 32; ++t) {  // the whole row: the softmax reads the winners of both halves
        noc_async_read(s_lg.get_noc_addr(t, roff), lg + t * 64, 32);
        noc_async_read(s_lg.get_noc_addr(t, roff + 512), lg + t * 64 + 32, 32);
    }
    noc_async_read(s_lg.get_noc_addr(num_experts / 32, roff), lg + num_experts * 2, 32);  // shared gate logit
    noc_async_read_barrier();
    invalidate_l1_cache();

    const uint32_t* words = reinterpret_cast<const uint32_t*>(lg);
    uint32_t keys[topk], ids[topk];
    moe_topk::select_range<topk>(words, 0, num_experts / 4, keys, ids);
    cb_wait_front(cb_part, 1);
    invalidate_l1_cache();
    const uint32_t* part = reinterpret_cast<const uint32_t*>(get_read_ptr(cb_part));
    moe_topk::merge<topk>(keys, ids, part, part + topk);
    cb_pop_front(cb_part, 1);
    uint16_t w[topk];
    moe_topk::softmax_selected<topk>(words, ids, w);

    volatile tt_l1_ptr uint16_t* r = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rec);
    for (uint32_t j = 0; j < 32; ++j) {
        r[j] = 0;
    }
    for (uint32_t j = 0; j < topk; ++j) {
        r[j] = static_cast<uint16_t>(ids[j]);
        r[17 + j] = w[j];
    }
    r[16] = moe_topk::sigmoid_bf16(reinterpret_cast<const uint16_t*>(lg)[num_experts]);
    noc_async_write(rec, s_table.get_noc_addr(row), 64);
    noc_async_write_barrier();
}
