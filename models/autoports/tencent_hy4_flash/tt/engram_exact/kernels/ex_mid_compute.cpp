// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// mid (FP32 DST): nw = n * w (row bcast, IEEE SFPU mul as ttnn.multiply); main = bf16(nw_0) (RNE, ttnn.typecast);
// kq_i = nw_{1+i} * nw_{5+i}.
#include "ex_common.hpp"

void kernel_main() {
    constexpr uint32_t J = get_compile_time_arg_val(0);
    constexpr uint32_t cb_n = 0, cb_w = 1, cb_main = 16, cb_kq = 17;
    compute_kernel_hw_startup(cb_n, cb_n, cb_main);
    cb_wait_front(cb_n, 9 * J);
    cb_wait_front(cb_w, 9 * J);
    for (uint32_t j = 0; j < J; ++j) {
        tile_regs_acquire();
        ex::copy_to(cb_n, j, 0);
        ex::bcast_row_to(cb_w, j, 1);
        ex::smul(0, 1, 0);
        ex::to_bf16(0);
        tile_regs_commit();
        tile_regs_wait();
        ex::pack_to(0, cb_main);
        tile_regs_release();
    }
    for (uint32_t i = 0; i < 4; ++i) {
        for (uint32_t j = 0; j < J; ++j) {
            const uint32_t tk = (1 + i) * J + j, tq = (5 + i) * J + j;
            tile_regs_acquire();
            ex::copy_to(cb_n, tk, 0);
            ex::bcast_row_to(cb_w, tk, 1);
            ex::smul(0, 1, 0);
            ex::copy_to(cb_n, tq, 2);
            ex::bcast_row_to(cb_w, tq, 3);
            ex::smul(2, 3, 2);
            ex::smul(0, 2, 0);
            tile_regs_commit();
            tile_regs_wait();
            ex::pack_to(0, cb_kq);
            tile_regs_release();
        }
    }
    cb_pop_front(cb_n, 9 * J);
    cb_pop_front(cb_w, 9 * J);
}
