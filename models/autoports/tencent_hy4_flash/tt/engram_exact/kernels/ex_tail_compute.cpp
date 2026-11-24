// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// tail (BF16 DST, as binary_ng's bf16 paths: ttnn.add -> FPU add_tiles (fast_and_approximate_mode defaults to true for
// ADD), ttnn.multiply -> SFPU mul_binary_tile on both operands copied to DST): coef = sc_i (row bcast) + g_i (col bcast);
// cm = coef * main; X_i' = X_i + cm. Same LLK ops / operand order / DST format as the three ttnn calls.
#include "ex_common.hpp"

void kernel_main() {
    constexpr uint32_t J = get_compile_time_arg_val(0);
    constexpr uint32_t DBG = get_compile_time_arg_val(1);  // 0: X'; 1: coef; 2: coef * main (debug)
    constexpr uint32_t cb_g = 0, cb_sc = 1, cb_m = 2, cb_x = 3, cb_gb = 4, cb_sb = 5, cb_coef = 6, cb_cm = 7,
                       cb_out = 16;
    compute_kernel_hw_startup(cb_sc, cb_g, cb_out);
    cb_wait_front(cb_g, 4);
    for (uint32_t i = 0; i < 4; ++i) {
        tile_regs_acquire();
        reconfig_data_format(cb_g, cb_g);
        unary_bcast_init<BroadcastType::COL>(cb_g);
        unary_bcast<BroadcastType::COL>(cb_g, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        ex::pack_to(0, cb_gb);
        tile_regs_release();
    }
    cb_pop_front(cb_g, 4);
    cb_wait_front(cb_gb, 4);
    cb_wait_front(cb_sc, 4 * J);
    cb_wait_front(cb_m, J);
    cb_wait_front(cb_x, 4 * J);
    for (uint32_t i = 0; i < 4; ++i) {
        for (uint32_t j = 0; j < J; ++j) {
            const uint32_t t = i * J + j;
            tile_regs_acquire();
            ex::bcast_row_to(cb_sc, t, 0);
            tile_regs_commit();
            tile_regs_wait();
            ex::pack_to(0, cb_sb);
            tile_regs_release();
            cb_wait_front(cb_sb, 1);

            tile_regs_acquire();
            reconfig_data_format(cb_sb, cb_gb);
            binary_tiles_init<true, EltwiseBinaryType::ELWADD>(cb_sb, cb_gb);
            add_tiles(cb_sb, cb_gb, 0, i, 0);
            tile_regs_commit();
            tile_regs_wait();
            ex::pack_to(0, DBG == 1 ? cb_out : cb_coef);
            tile_regs_release();
            cb_pop_front(cb_sb, 1);
            if constexpr (DBG == 1) {
                continue;
            }
            cb_wait_front(cb_coef, 1);

            tile_regs_acquire();
            ex::copy_to(cb_coef, 0, 0);
            ex::copy_to(cb_m, j, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            ex::pack_to(0, DBG == 2 ? cb_out : cb_cm);
            tile_regs_release();
            cb_pop_front(cb_coef, 1);
            if constexpr (DBG == 2) {
                continue;
            }
            cb_wait_front(cb_cm, 1);

            tile_regs_acquire();
            reconfig_data_format(cb_x, cb_cm);
            binary_tiles_init<true, EltwiseBinaryType::ELWADD>(cb_x, cb_cm);
            add_tiles(cb_x, cb_cm, t, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            ex::pack_to(0, cb_out);
            tile_regs_release();
            cb_pop_front(cb_cm, 1);
        }
    }
    cb_pop_front(cb_gb, 4);
    cb_pop_front(cb_sc, 4 * J);
    cb_pop_front(cb_m, J);
    cb_pop_front(cb_x, 4 * J);
}
