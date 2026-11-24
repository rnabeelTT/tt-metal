// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// gate (FP32 DST), one stream's tile per core: g = bf16(sigmoid(sign(s c) sqrt(max(|s c|, 1e-6)))), s = the row sums,
// c = 1 / sqrt(H): the ttnn ops of the reference chain (multiply, abs, clamp, sqrt, sign, multiply, sigmoid,
// typecast), each IEEE / the same SFPU function.
#include "ex_common.hpp"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/sqrt.h"

void kernel_main() {
    constexpr uint32_t isq_bits = get_compile_time_arg_val(0);
    constexpr uint32_t gmin_bits = get_compile_time_arg_val(1);
    constexpr uint32_t cb_s = 0, cb_g = 16;
    compute_kernel_hw_startup(cb_s, cb_s, cb_g);
    cb_wait_front(cb_s, 1);
    tile_regs_acquire();
    ex::copy_to(cb_s, 0, 0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, isq_bits);
    copy_dest_values_init();
    copy_dest_values<DataFormat::Float32>(0, 1);
    abs_tile_init();
    abs_tile(1);
    unary_max_tile_init();
    unary_max_tile(1, gmin_bits);
    sqrt_tile_init();
    sqrt_tile(1);
    sign_tile_init();
    sign_tile(0);
    ex::smul(1, 0, 1);
    sigmoid_tile_init();
    sigmoid_tile(1);
    ex::to_bf16(1);
    tile_regs_commit();
    tile_regs_wait();
    ex::pack_to(1, cb_g);
    tile_regs_release();
    cb_pop_front(cb_s, 1);
}
