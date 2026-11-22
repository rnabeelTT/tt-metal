// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Batched short conv step (tt/bconv.py), compute: per channel tile, in FP32 DST (unpack-to-dest FP32, SFPU math):
//   y = x * wc + s0 * w0 + s1 * w1 + s2 * w2        -> cb_y (bf16)
//   the rotated states: s1 -> cb_o0, s2 -> cb_o1, x (as fp32) -> cb_o2 (the writer stores them as s0, s1, s2)
// DST tiles: 0 = x, 1 / 2 = operands, 4 = y accumulator (full-sync DST: 8 FP32 tiles).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_s = get_compile_time_arg_val(1);
    constexpr uint32_t cb_w = get_compile_time_arg_val(2);
    constexpr uint32_t cb_y = get_compile_time_arg_val(3);
    constexpr uint32_t cb_o0 = get_compile_time_arg_val(4);
    constexpr uint32_t cb_o1 = get_compile_time_arg_val(5);
    constexpr uint32_t cb_o2 = get_compile_time_arg_val(6);
    const uint32_t nt = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_s, cb_y);
    for (uint32_t t = 0; t < nt; ++t) {
        cb_wait_front(cb_x, 1);
        cb_wait_front(cb_s, 3);
        cb_wait_front(cb_w, 4);
        tile_regs_acquire();
        // y = x * wc
        reconfig_data_format_srca(cb_x);
        copy_init(cb_x);
        copy_tile(cb_x, 0, 0);
        reconfig_data_format_srca(cb_w);
        copy_init(cb_w);
        copy_tile(cb_w, 3, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 4);
        // y += s_j * w_j
        for (uint32_t j = 0; j < 3; ++j) {
            reconfig_data_format_srca(cb_s);
            copy_init(cb_s);
            copy_tile(cb_s, j, 1);
            reconfig_data_format_srca(cb_w);
            copy_init(cb_w);
            copy_tile(cb_w, j, 2);
            mul_binary_tile_init();
            mul_binary_tile(1, 2, 1);
            add_binary_tile_init();
            add_binary_tile(4, 1, 4);
        }
        // the rotated states (exact FP32 copies through DST)
        reconfig_data_format_srca(cb_s);
        copy_init(cb_s);
        copy_tile(cb_s, 1, 1);
        copy_tile(cb_s, 2, 2);
        tile_regs_commit();
        cb_reserve_back(cb_y, 1);
        cb_reserve_back(cb_o0, 1);
        cb_reserve_back(cb_o1, 1);
        cb_reserve_back(cb_o2, 1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_y);
        pack_tile(4, cb_y);
        pack_reconfig_data_format(cb_o0);
        pack_tile(1, cb_o0);
        pack_tile(2, cb_o1);
        pack_tile(0, cb_o2);
        tile_regs_release();
        cb_push_back(cb_y, 1);
        cb_push_back(cb_o0, 1);
        cb_push_back(cb_o1, 1);
        cb_push_back(cb_o2, 1);
        cb_pop_front(cb_x, 1);
        cb_pop_front(cb_s, 3);
        cb_pop_front(cb_w, 4);
    }
}
