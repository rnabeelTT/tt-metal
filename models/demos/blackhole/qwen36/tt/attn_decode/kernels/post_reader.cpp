// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Attention decode post, reader: user b's Ht tiles of the SDPA output and of the gate (pages b * Ht + j).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t g_addr = get_arg_val<uint32_t>(1);
    const uint32_t start = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t cb_a = get_compile_time_arg_val(1);
    constexpr uint32_t cb_g = get_compile_time_arg_val(2);
    constexpr auto a_args = TensorAccessorArgs<3>();
    constexpr auto g_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr uint32_t BF_TILE = 2048;
    const auto s_a = TensorAccessor(a_args, a_addr, BF_TILE);
    const auto s_g = TensorAccessor(g_args, g_addr, BF_TILE);
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t b = start + i;
        cb_reserve_back(cb_a, Ht);
        cb_reserve_back(cb_g, Ht);
        const uint32_t a = get_write_ptr(cb_a), g = get_write_ptr(cb_g);
        for (uint32_t j = 0; j < Ht; ++j) {
            noc_async_read(s_a.get_noc_addr(b * Ht + j), a + j * BF_TILE, BF_TILE);
            noc_async_read(s_g.get_noc_addr(b * Ht + j), g + j * BF_TILE, BF_TILE);
        }
        noc_async_read_barrier();
        cb_push_back(cb_a, Ht);
        cb_push_back(cb_g, Ht);
    }
}
