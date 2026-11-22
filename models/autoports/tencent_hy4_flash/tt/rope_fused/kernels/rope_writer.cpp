// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fused RoPE, writer (RISCV_0): y tiles [t0, t1) (roped tile t: x tile row t / 2, column tile WT - 2 + t % 2). KEEP: y
// is x's shape and the core handling a row's first roped tile also copies that row's WT - 2 leading tiles from x.
// runtime args: [y, t0, t1, x]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t t0 = get_arg_val<uint32_t>(1);
    const uint32_t t1 = get_arg_val<uint32_t>(2);
    constexpr uint32_t WT = get_compile_time_arg_val(0);
    constexpr uint32_t KEEP = get_compile_time_arg_val(1);
    constexpr uint32_t cb_y = 16, cb_c = 17;
    constexpr auto y_args = TensorAccessorArgs<2>();
    constexpr auto x_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr uint32_t BF = 2048;
    const auto s_y = TensorAccessor(y_args, y_addr, BF);
    if constexpr (KEEP) {
        const auto s_x = TensorAccessor(x_args, get_arg_val<uint32_t>(3), BF);
        cb_reserve_back(cb_c, 1);
        const uint32_t buf = get_write_ptr(cb_c);
        for (uint32_t t = t0; t < t1; ++t) {
            if (t & 1) {
                continue;
            }
            const uint32_t row = t >> 1;
            for (uint32_t k = 0; k < WT - 2; ++k) {
                noc_async_read(s_x.get_noc_addr(row * WT + k), buf, BF);
                noc_async_read_barrier();
                noc_async_write(buf, s_y.get_noc_addr(row * WT + k), BF);
                noc_async_write_barrier();
            }
        }
    }
    for (uint32_t t = t0; t < t1; ++t) {
        cb_wait_front(cb_y, 1);
        const uint32_t page = KEEP ? (t >> 1) * WT + WT - 2 + (t & 1) : t;
        noc_async_write(get_read_ptr(cb_y), s_y.get_noc_addr(page), BF);
        noc_async_write_barrier();
        cb_pop_front(cb_y, 1);
    }
}
