// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// GDN decode short conv, compute: out = silu(t0 h0 + t1 h1 + t2 h2 + t3 x) per tile column, on the SFPU in FP32 DST
// (bf16 operands unpack exactly), packed FP32 for the head kernel.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_h = get_compile_time_arg_val(1);
    constexpr uint32_t cb_t = get_compile_time_arg_val(2);
    constexpr uint32_t cb_o = get_compile_time_arg_val(3);
    const uint32_t n = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_h, cb_t, cb_o);
    for (uint32_t i = 0; i < n; ++i) {
        cb_wait_front(cb_h, 3);
        cb_wait_front(cb_x, 1);
        cb_wait_front(cb_t, 4);
        cb_reserve_back(cb_o, 1);
        tile_regs_acquire();
        copy_init(cb_h);
        copy_tile(cb_h, 0, 0);
        copy_init(cb_t);
        copy_tile(cb_t, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        for (uint32_t j = 1; j < 4; ++j) {
            if (j < 3) {
                copy_init(cb_h);
                copy_tile(cb_h, j, 1);
            } else {
                copy_init(cb_x);
                copy_tile(cb_x, 0, 1);
            }
            copy_init(cb_t);
            copy_tile(cb_t, j, 2);
            mul_binary_tile_init();
            mul_binary_tile(1, 2, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        }
        silu_tile_init();
        silu_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_o);
        tile_regs_release();
        cb_push_back(cb_o, 1);
        cb_pop_front(cb_h, 3);
        cb_pop_front(cb_x, 1);
        cb_pop_front(cb_t, 4);
    }
}
