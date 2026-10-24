// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Attention decode post, writer: row h of product tile j -> row b of tile h * Ht + j of the out-proj input
// ([1, B, NH * HD] TILE): two 32-byte face-row writes, disjoint from every other user's.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);
    constexpr uint32_t cb_y = get_compile_time_arg_val(2);
    constexpr auto y_args = TensorAccessorArgs<3>();
    constexpr uint32_t BF_TILE = 2048, BF_FACE = 512, BF_ROW = 32;
    const auto s_y = TensorAccessor(y_args, y_addr, BF_TILE);
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t b = start + i;
        const uint32_t r = b & 31;
        cb_wait_front(cb_y, Ht);
        const uint32_t src = get_read_ptr(cb_y);
        for (uint32_t h = 0; h < NH; ++h) {
            for (uint32_t j = 0; j < Ht; ++j) {
                const uint32_t page = (b >> 5) * (NH * Ht) + h * Ht + j;
                for (uint32_t c = 0; c < 2; ++c) {
                    const uint32_t src_off = ((h >> 4) * 2 + c) * BF_FACE + (h & 15) * BF_ROW;
                    const uint32_t dst_off = ((r >> 4) * 2 + c) * BF_FACE + (r & 15) * BF_ROW;
                    noc_async_write(src + j * BF_TILE + src_off, s_y.get_noc_addr(page, dst_off), BF_ROW);
                }
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_y, Ht);
    }
}
