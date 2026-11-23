// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// DSpark Markov step logits (tt/drafter.py MarkovStep): per output column tile, [lat, 1] @ [fold W; fold b] (kt K tiles,
// FP32 DST) + the column's base logits tile (FP32, unpacked to DST, SFPU add), packed FP32 for the writer's argmax.
//
// runtime args: [ncols]      compile args: cb_x, cb_w, cb_base, cb_out, kt

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_base = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t kt = get_compile_time_arg_val(4);
    const uint32_t ncols = get_arg_val<uint32_t>(0);
    if (ncols == 0) {
        return;
    }
    compute_kernel_hw_startup(cb_x, cb_w, cb_out);
    cb_wait_front(cb_x, kt);
    for (uint32_t j = 0; j < ncols; ++j) {
        reconfig_data_format(cb_w, cb_x);  // matmul: in1 (weights) -> srcA, in0 -> srcB
        matmul_init(cb_x, cb_w, 0);
        cb_wait_front(cb_w, kt);
        cb_wait_front(cb_base, 1);
        tile_regs_acquire();
        for (uint32_t k = 0; k < kt; ++k) {
            matmul_tiles(cb_x, cb_w, k, k, 0);
        }
        reconfig_data_format_srca(cb_base);
        copy_tile_to_dst_init_short(cb_base);
        copy_tile(cb_base, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        tile_regs_commit();
        cb_pop_front(cb_w, kt);
        cb_pop_front(cb_base, 1);
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
    cb_pop_front(cb_x, kt);
}
