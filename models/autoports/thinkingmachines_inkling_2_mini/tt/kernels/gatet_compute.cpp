// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Message gate tail (tt/gatet.py), compute (fp32 DST, SFPU): out = h - tanh(f) * h + tanh(a) * m
// DST: 0 = f, 1 = a, 2 = h, 3 = m.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    const uint32_t nt = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_in, cb_out);
    for (uint32_t t = 0; t < nt; ++t) {
        cb_wait_front(cb_in, 4);
        tile_regs_acquire();
        copy_init(cb_in);
        for (uint32_t i = 0; i < 4; ++i) {
            copy_tile(cb_in, i, i);
        }
        tanh_tile_init();
        tanh_tile(0);
        tanh_tile(1);
        mul_binary_tile_init();
        mul_binary_tile(0, 2, 0);  // tanh(f) h
        mul_binary_tile(1, 3, 1);  // tanh(a) m
        sub_binary_tile_init();
        sub_binary_tile(2, 0, 2);  // h - tanh(f) h
        add_binary_tile_init();
        add_binary_tile(2, 1, 2);
        tile_regs_commit();
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(2, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_in, 4);
    }
}
