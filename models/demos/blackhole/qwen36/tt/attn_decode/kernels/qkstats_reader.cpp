// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Full-width QK-norm statistics, reader: this core's cnt tiles of the [1, 1, B <= 32, W] TILE bf16 projection (one
// tile row: page = tile column), starting at column c0, plus the all-ones tile (matmul operand: row sums).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t qkv_addr = get_arg_val<uint32_t>(0);
    const uint32_t ones_addr = get_arg_val<uint32_t>(1);
    const uint32_t c0 = get_arg_val<uint32_t>(2);
    const uint32_t cnt = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(1);
    constexpr auto qkv_args = TensorAccessorArgs<2>();
    constexpr auto ones_args = TensorAccessorArgs<qkv_args.next_compile_time_args_offset()>();
    constexpr uint32_t BF_TILE = 2048;
    const auto s_qkv = TensorAccessor(qkv_args, qkv_addr, BF_TILE);
    const auto s_ones = TensorAccessor(ones_args, ones_addr, BF_TILE);

    cb_reserve_back(cb_ones, 1);
    noc_async_read(s_ones.get_noc_addr(0), get_write_ptr(cb_ones), BF_TILE);
    cb_reserve_back(cb_x, cnt);
    uint32_t x = get_write_ptr(cb_x);
    for (uint32_t j = 0; j < cnt; ++j) {
        noc_async_read(s_qkv.get_noc_addr(c0 + j), x, BF_TILE);
        x += BF_TILE;
    }
    noc_async_read_barrier();
    cb_push_back(cb_ones, 1);
    cb_push_back(cb_x, cnt);
}
