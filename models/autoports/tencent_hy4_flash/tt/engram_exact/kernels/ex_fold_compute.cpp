// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// fold (FP32 DST): S = ((t_0 + t_1) + t_2) + ... + t_{HT-1}, elementwise IEEE adds in tile order: the cross-tile part
// of ttnn.sum(fp32, dim=-1)'s accurate SFPU reduce (compute_kernel_lib::reduce: sfpu_copy_and_fold); the in-tile row
// sum is left to ttnn.sum on the one folded tile. ct: [HT, BLK]
#include "ex_common.hpp"

void kernel_main() {
    constexpr uint32_t HT = get_compile_time_arg_val(0);
    constexpr uint32_t BLK = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in = 0, cb_out = 16;
    compute_kernel_hw_startup(cb_in, cb_in, cb_out);
    tile_regs_acquire();
    reconfig_data_format_srca(cb_in);
    copy_init(cb_in);
    for (uint32_t t0 = 0; t0 < HT; t0 += BLK) {
        cb_wait_front(cb_in, BLK);
        for (uint32_t k = 0; k < BLK; ++k) {
            if (t0 + k == 0) {
                copy_tile(cb_in, k, 0);
            } else {
                copy_tile(cb_in, k, 1);
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
                copy_init(cb_in);
            }
        }
        cb_pop_front(cb_in, BLK);
    }
    tile_regs_commit();
    tile_regs_wait();
    ex::pack_to(0, cb_out);
    tile_regs_release();
}
