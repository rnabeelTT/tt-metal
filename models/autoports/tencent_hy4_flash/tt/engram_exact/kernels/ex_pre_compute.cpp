// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// pre (FP32 DST): allv = [fp32(bf16(y_s + b_s)) s = v, k_0..3 ; fp32(X_i)], this core's J hidden tiles of each.
// Same SFPU add (IEEE) and RNE rounding as ttnn.add(y, b, dtype=bf16) -> typecast; concat.
#include "ex_common.hpp"

void kernel_main() {
    constexpr uint32_t J = get_compile_time_arg_val(0);
    constexpr uint32_t cb_y = 0, cb_yb = 1, cb_x = 2, cb_out = 16;
    compute_kernel_hw_startup(cb_y, cb_y, cb_out);
    cb_wait_front(cb_y, 5 * J);
    cb_wait_front(cb_yb, 5 * J);
    for (uint32_t t = 0; t < 5 * J; ++t) {
        tile_regs_acquire();
        ex::copy_to(cb_y, t, 0);
        ex::bcast_row_to(cb_yb, t, 1);
        ex::sadd(0, 1, 0);
        ex::to_bf16(0);
        tile_regs_commit();
        tile_regs_wait();
        ex::pack_to(0, cb_out);
        tile_regs_release();
    }
    cb_pop_front(cb_y, 5 * J);
    cb_pop_front(cb_yb, 5 * J);
    cb_wait_front(cb_x, 4 * J);
    for (uint32_t t = 0; t < 4 * J; ++t) {
        tile_regs_acquire();
        ex::copy_to(cb_x, t, 0);
        tile_regs_commit();
        tile_regs_wait();
        ex::pack_to(0, cb_out);
        tile_regs_release();
    }
    cb_pop_front(cb_x, 4 * J);
}
