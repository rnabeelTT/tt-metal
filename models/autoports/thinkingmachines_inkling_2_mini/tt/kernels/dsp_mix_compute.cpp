// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// DSpark drafter tap mixing, compute (tt/drafter.py _mix): per tile, mix m = ((a_m0 T0 + a_m1 T1) + a_m2 T2) + a_m3 T3
// in FP32 DST (the taps bf16, exact), packed FP32. Compile args: cb_in, cb_out, NM (<= 4), then 16 FP32 bits a_mj.
//
// runtime args: [nt]

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t NM = get_compile_time_arg_val(2);
    constexpr uint32_t A[16] = {
        get_compile_time_arg_val(3),  get_compile_time_arg_val(4),  get_compile_time_arg_val(5),
        get_compile_time_arg_val(6),  get_compile_time_arg_val(7),  get_compile_time_arg_val(8),
        get_compile_time_arg_val(9),  get_compile_time_arg_val(10), get_compile_time_arg_val(11),
        get_compile_time_arg_val(12), get_compile_time_arg_val(13), get_compile_time_arg_val(14),
        get_compile_time_arg_val(15), get_compile_time_arg_val(16), get_compile_time_arg_val(17),
        get_compile_time_arg_val(18)};
    const uint32_t nt = get_arg_val<uint32_t>(0);
    if (nt == 0) {
        return;
    }
    compute_kernel_hw_startup(cb_in, cb_out);
    for (uint32_t i = 0; i < nt; ++i) {
        cb_wait_front(cb_in, 4);
        cb_reserve_back(cb_out, NM);
        for (uint32_t m = 0; m < NM; ++m) {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(cb_in);
            copy_tile(cb_in, 0, 0);
            binop_with_scalar_tile_init();
            mul_unary_tile(0, A[m * 4]);
            for (uint32_t j = 1; j < 4; ++j) {
                copy_tile_to_dst_init_short(cb_in);
                copy_tile(cb_in, j, 1);
                binop_with_scalar_tile_init();
                mul_unary_tile(1, A[m * 4 + j]);
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
        }
        cb_push_back(cb_out, NM);
        cb_pop_front(cb_in, 4);
    }
}
