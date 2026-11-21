// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Inkling short conv over a prefill chunk, compute (pf_reader.cpp's work split): per row tile of the core's range,
// y = x + t0 T0 + t1 T1 + t2 T2 + t3 x in FP32 DST on the SFPU, in the reference's summation order (compute.cpp's
// arithmetic), the 4 tap tiles held for the whole range. The fp32 sum is rounded to bf16 on the SFPU (round to nearest
// even, as ttnn.typecast) before the pack: the packer's own fp32 -> bf16 conversion differs in ~0.1% of values.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_s = get_compile_time_arg_val(1);
    constexpr uint32_t cb_t = get_compile_time_arg_val(2);
    constexpr uint32_t cb_o = get_compile_time_arg_val(3);
    constexpr uint32_t cols = get_compile_time_arg_val(4);
    constexpr uint32_t Rt = get_compile_time_arg_val(5);
    constexpr uint32_t G = get_compile_time_arg_val(6);
    constexpr uint32_t GY = get_compile_time_arg_val(7);

    const uint32_t item = (uint32_t)get_absolute_logical_x() * GY + (uint32_t)get_absolute_logical_y();
    if (item >= cols * G) {
        return;
    }
    constexpr uint32_t per = (Rt + G - 1) / G;
    const uint32_t k0 = (item / cols) * per;
    const uint32_t k1 = (k0 + per < Rt) ? k0 + per : Rt;
    const uint32_t n = k1 > k0 ? k1 - k0 : 0;

    compute_kernel_hw_startup(cb_x, cb_t, cb_o);
    cb_wait_front(cb_t, 4);
    for (uint32_t i = 0; i < n; ++i) {
        cb_wait_front(cb_x, 1);
        cb_wait_front(cb_s, 3);
        cb_reserve_back(cb_o, 1);
        tile_regs_acquire();
        copy_init(cb_x);
        copy_tile(cb_x, 0, 0);
        for (uint32_t j = 0; j < 4; ++j) {
            if (j < 3) {
                copy_init(cb_s);
                copy_tile(cb_s, j, 1);
            } else {
                copy_init(cb_x);
                copy_tile(cb_x, 0, 1);
            }
            copy_init(cb_t);
            copy_tile(cb_t, j, 2);
            mul_binary_tile_init();
            mul_binary_tile(1, 2, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        }
        typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>();
        typecast_tile<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_o);
        tile_regs_release();
        cb_push_back(cb_o, 1);
        cb_pop_front(cb_x, 1);
        cb_pop_front(cb_s, 3);
    }
    cb_pop_front(cb_t, 4);
}
