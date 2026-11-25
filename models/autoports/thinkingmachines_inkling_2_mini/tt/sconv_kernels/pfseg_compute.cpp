// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Inkling short conv over a packed prefill chunk, compute (pfseg_reader.cpp's work split): per row tile k,
//   T_j = P_j x_k + Q_j p_k   (j < 3: the rows shifted down by 3 - j; p_k = the 32 rows before x_k: x_{k-1} or the
//                              segment's state rows at 29 .. 31; P_j / Q_j 0/1 shift matrices, exact on the FPU)
//   y = x + t0 T0 + t1 T1 + t2 T2 + t3 x   (fp32 DST on the SFPU, pf_compute.cpp's arithmetic and order)
// rounded to bf16 on the SFPU (round to nearest even) before the pack. PF_RESID: out = r + y (r the residual tile,
// fp32 in DST; = ttnn.add(f32(r), f32(y)) after the conv), rounded to bf16 again when PF_OUT_BF16 (the output CB's
// format: fp32 or bf16).
// DST (full sync, 8 fp32 tiles): 0 = y, 1 .. 3 = T_j, 4 = tap, 5 = x, 6 = r.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_p = get_compile_time_arg_val(1);
    constexpr uint32_t cb_t = get_compile_time_arg_val(2);
    constexpr uint32_t cb_o = get_compile_time_arg_val(3);
    constexpr uint32_t cols = get_compile_time_arg_val(4);
    constexpr uint32_t Rt = get_compile_time_arg_val(5);
    constexpr uint32_t G = get_compile_time_arg_val(6);
    constexpr uint32_t GY = get_compile_time_arg_val(7);
    constexpr uint32_t cb_m = get_compile_time_arg_val(8);  // 6 tiles: P_0 .. P_2, Q_0 .. Q_2
    constexpr uint32_t cb_r = get_compile_time_arg_val(9);  // PF_RESID: the residual tiles

    const uint32_t item = (uint32_t)get_absolute_logical_x() * GY + (uint32_t)get_absolute_logical_y();
#ifdef PF_FLAT
    if (item >= G) {
        return;
    }
    const uint32_t t0 = item * (cols * Rt) / G;
    const uint32_t n = (item + 1) * (cols * Rt) / G - t0;
#else
    if (item >= cols * G) {
        return;
    }
    constexpr uint32_t per = (Rt + G - 1) / G;
    const uint32_t k0 = (item / cols) * per;
    const uint32_t k1 = (k0 + per < Rt) ? k0 + per : Rt;
    const uint32_t n = k1 > k0 ? k1 - k0 : 0;
#endif

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_m, cb_x, cb_o);  // (matmul: in1 -> srcA)
#ifndef PF_FLAT
    cb_wait_front(cb_t, 4);
#endif
    cb_wait_front(cb_m, 6);
    for (uint32_t i = 0; i < n; ++i) {
#ifdef PF_FLAT
        if (i == 0 || (t0 + i) % Rt == 0) {  // a new column: its taps
            if (i > 0) {
                cb_pop_front(cb_t, 4);
            }
            cb_wait_front(cb_t, 4);
        }
#endif
        cb_wait_front(cb_x, 1);
        cb_wait_front(cb_p, 1);
        cb_reserve_back(cb_o, 1);
        tile_regs_acquire();
        reconfig_data_format(cb_x, cb_m);
        matmul_init(cb_m, cb_x, 0);
        for (uint32_t j = 0; j < 3; ++j) {
            matmul_tiles(cb_m, cb_x, j, 0, 1 + j);
            matmul_tiles(cb_m, cb_p, 3 + j, 0, 1 + j);
        }
        reconfig_data_format_srca(cb_x);
        copy_init(cb_x);
        copy_tile(cb_x, 0, 0);
        copy_tile(cb_x, 0, 5);
        for (uint32_t j = 0; j < 4; ++j) {
            copy_init(cb_t);
            copy_tile(cb_t, j, 4);
            mul_binary_tile_init();
            mul_binary_tile(j < 3 ? 1 + j : 5, 4, j < 3 ? 1 + j : 5);
            add_binary_tile_init();
            add_binary_tile(0, j < 3 ? 1 + j : 5, 0);
        }
        typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>();
        typecast_tile<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>(0);
#ifdef PF_RESID
        cb_wait_front(cb_r, 1);
        reconfig_data_format_srca(cb_r);
        copy_init(cb_r);
        copy_tile(cb_r, 0, 6);
        add_binary_tile_init();
        add_binary_tile(6, 0, 0);
#if PF_OUT_BF16
        typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>();
        typecast_tile<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>(0);
#endif
        cb_pop_front(cb_r, 1);
#endif
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_o);
        pack_tile(0, cb_o);
        tile_regs_release();
        cb_push_back(cb_o, 1);
        cb_pop_front(cb_x, 1);
        cb_pop_front(cb_p, 1);
    }
    if (n > 0) {
        cb_pop_front(cb_t, 4);
    }
    cb_pop_front(cb_m, 6);
}
