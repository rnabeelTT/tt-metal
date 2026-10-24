// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Attention decode prep, writer: an item's Ht result tiles -> pages b * Ht + j of its output ([1, B, heads, HD] TILE,
// one tile row per user); k items also forward v's tiles from the reader.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t g_addr = get_arg_val<uint32_t>(3);
    const uint32_t start = get_arg_val<uint32_t>(4);
    const uint32_t count = get_arg_val<uint32_t>(5);

    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t cb_v = get_compile_time_arg_val(2);
    constexpr auto q_args = TensorAccessorArgs<3>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr uint32_t BF_TILE = 2048;
    const auto s_q = TensorAccessor(q_args, q_addr, BF_TILE);
    const auto s_k = TensorAccessor(k_args, k_addr, BF_TILE);
    const auto s_v = TensorAccessor(v_args, v_addr, BF_TILE);
    const auto s_g = TensorAccessor(g_args, g_addr, BF_TILE);

    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t u = start + i;
        const uint32_t b = u / 3;
        const uint32_t part = u - 3 * b;
        cb_wait_front(cb_out, Ht);
        DeviceZoneScopedN("wr_item");
        const uint32_t o = get_read_ptr(cb_out);
        for (uint32_t j = 0; j < Ht; ++j) {
            const uint32_t page = b * Ht + j;
            const uint64_t dst = part == 0 ? s_q.get_noc_addr(page) : part == 1 ? s_k.get_noc_addr(page) : s_g.get_noc_addr(page);
            noc_async_write(o + j * BF_TILE, dst, BF_TILE);
        }
        if (part == 1) {
            cb_wait_front(cb_v, Ht);
            const uint32_t v = get_read_ptr(cb_v);
            for (uint32_t j = 0; j < Ht; ++j) {
                noc_async_write(v + j * BF_TILE, s_v.get_noc_addr(b * Ht + j), BF_TILE);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, Ht);
        if (part == 1) {
            cb_pop_front(cb_v, Ht);
        }
    }
}
