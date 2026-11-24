// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Router post, compute (FP32 DST, FP32 unpack-to-dest copies): s = sigmoid(a + b); key = s + bias.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_a = 0, cb_b = 1, cb_c = 2, cb_k = 16, cb_s = 17;
    const uint32_t n = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_a, cb_a, cb_k);
    pack_reconfig_data_format(cb_k);
    for (uint32_t t = 0; t < n; ++t) {
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_b, 1);
        cb_wait_front(cb_c, 1);
        cb_reserve_back(cb_k, 1);
        cb_reserve_back(cb_s, 1);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_a);
        copy_init(cb_a);
        copy_tile(cb_a, 0, 0);
        reconfig_data_format_srca(cb_b);
        copy_init(cb_b);
        copy_tile(cb_b, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);  // logits = a + b
        sigmoid_tile_init();
        sigmoid_tile(0);  // s
        reconfig_data_format_srca(cb_c);
        copy_init(cb_c);
        copy_tile(cb_c, 0, 1);
        add_binary_tile_init();
        add_binary_tile(1, 0, 1);  // key = bias + s
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(1, cb_k);
        pack_tile(0, cb_s);
        tile_regs_release();
        cb_push_back(cb_k, 1);
        cb_push_back(cb_s, 1);
        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_b, 1);
        cb_pop_front(cb_c, 1);
    }
}
