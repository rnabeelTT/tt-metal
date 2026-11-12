// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Full-width QK-norm statistics, compute: the squares of this core's cnt tiles (FP32), then one matmul chain with the
// all-ones tile: every column of the result holds the row's sum of squares over the cnt tiles (FP32 accumulation).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(1);
    constexpr uint32_t cb_sq = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    const uint32_t cnt = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_x, cb_x, cb_sq);
    cb_wait_front(cb_x, cnt);
    cb_wait_front(cb_ones, 1);
    cb_reserve_back(cb_sq, cnt);
    pack_reconfig_data_format(cb_sq);
    reconfig_data_format(cb_x, cb_x);
    mul_init(cb_x, cb_x);
    for (uint32_t j0 = 0; j0 < cnt; j0 += 4) {
        tile_regs_acquire();
        for (uint32_t j = j0; j < j0 + 4 && j < cnt; j++) {
            mul_tiles(cb_x, cb_x, j, j, j - j0);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = j0; j < j0 + 4 && j < cnt; j++) {
            pack_tile(j - j0, cb_sq);
        }
        tile_regs_release();
    }
    cb_push_back(cb_sq, cnt);
    cb_pop_front(cb_x, cnt);

    cb_wait_front(cb_sq, cnt);
    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();
    reconfig_data_format(cb_ones, cb_sq);  // matmul: in1 -> srcA, in0 -> srcB
    matmul_init(cb_sq, cb_ones, 0);
    for (uint32_t j = 0; j < cnt; j++) {
        matmul_tiles(cb_sq, cb_ones, j, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_out);
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
    cb_pop_front(cb_sq, cnt);
    cb_pop_front(cb_ones, 1);
}
