// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Inkling LM head, compute (tt/head.py StreamHead; the qwen36 DecodeLinear reader streams this core's columns): per
// output column, sum over the kt K tiles of x[k] @ W[k, column] (FP32 DST), then the softcap c tanh(y / c) on the
// SFPU (HEAD_SOFTCAP; 1/c and c as FP32 bits), packed FP32.
//
// runtime args: [ncols]      compile args: cb_x, cb_w, cb_out, kt, 1/c bits, c bits

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t kt = get_compile_time_arg_val(3);
    constexpr uint32_t inv_c = get_compile_time_arg_val(4);
    constexpr uint32_t c = get_compile_time_arg_val(5);
    const uint32_t ncols = get_arg_val<uint32_t>(0);
    if (ncols == 0) {
        return;
    }
    compute_kernel_hw_startup(cb_x, cb_w, cb_out);
    reconfig_data_format(cb_w, cb_x);  // matmul: in1 (weights) -> srcA, in0 -> srcB
    cb_wait_front(cb_x, kt);
    for (uint32_t j = 0; j < ncols; ++j) {
        matmul_init(cb_x, cb_w, 0);
        cb_wait_front(cb_w, kt);
        tile_regs_acquire();
        for (uint32_t k = 0; k < kt; ++k) {
            matmul_tiles(cb_x, cb_w, k, k, 0);
        }
#ifdef HEAD_SOFTCAP
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv_c);
        tanh_tile_init();
        tanh_tile(0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, c);
#endif
        tile_regs_commit();
        cb_pop_front(cb_w, kt);
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
    cb_pop_front(cb_x, kt);
}
