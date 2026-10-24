// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Attention decode prep, compute. Item u = 3 b + part; the heads of the item are the rows of Ht bf16 tiles (cb_x):
//   part 0 / 1 (q / k): r = 1 / sqrt(mean_d x^2 + eps) per row (squares, then a matmul with the all-ones column tile
//     summing each row over the Ht tiles), x r (column broadcast) (1 + w) (row broadcast), then the HF rotate-half RoPE
//     on tiles 0 / 1 (the first rope_dim = 64 dims): y0 = x0 cos0 - x1 sin0, y1 = x1 cos1 + x0 sin1 (row broadcasts);
//   part 2 (gate): sigmoid.
// Results bf16 to cb_out in tile order.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

namespace {
constexpr uint32_t cb_x = 0, cb_w = 1, cb_cs = 2, cb_ones = 3, cb_sq = 4, cb_r = 5, cb_xn = 6, cb_out = 7;

ALWI void copy_from(uint32_t cb) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
}

// FPU work after SFPU work in one acquire: make the FPU wait for the SFPU
ALWI void sfpu_drain() {
#ifdef TRISC_MATH
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
#endif
}

template <uint32_t Ht>
void norm_rope(uint32_t wofs, uint32_t eps_bits, uint32_t inv_d_bits) {
    CircularBuffer cx(cb_x), csq(cb_sq), cr(cb_r), cxn(cb_xn), co(cb_out), ccs(cb_cs);
    cx.wait_front(Ht);
    // squares (FP32)
    csq.reserve_back(Ht);
    pack_reconfig_data_format(cb_sq);
    copy_from(cb_x);
    mul_binary_tile_init();
    for (uint32_t j = 0; j < Ht; j++) {
        tile_regs_acquire();
        copy_tile(cb_x, j, 0);
        mul_binary_tile(0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_sq);
        tile_regs_release();
    }
    csq.push_back(Ht);
    // r = rsqrt(sum / D + eps), column 0 (per row)
    csq.wait_front(Ht);
    cr.reserve_back(1);
    tile_regs_acquire();
    reconfig_data_format(cb_ones, cb_sq);  // matmul: in1 -> srcA, in0 -> srcB
    matmul_init(cb_sq, cb_ones, 0);
    for (uint32_t j = 0; j < Ht; j++) {
        matmul_tiles(cb_sq, cb_ones, j, 0, 0);
    }
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_d_bits);
    add_unary_tile(0, eps_bits);
    rsqrt_tile_init<false>();
    rsqrt_tile<false, false>(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_r);
    tile_regs_release();
    cr.push_back(1);
    csq.pop_front(Ht);
    cr.wait_front(1);
    // x r (1 + w): tiles 0 / 1 -> cb_xn (FP32, for the RoPE), the rest -> cb_out (bf16)
    co.reserve_back(Ht);
    cxn.reserve_back(2);
    for (uint32_t j = 0; j < Ht; j++) {
        if (j == 0) {
            pack_reconfig_data_format(cb_xn);
        } else if (j == 2) {
            pack_reconfig_data_format(cb_out);
        }
        tile_regs_acquire();
        copy_from(cb_x);
        copy_tile(cb_x, j, 0);
        copy_from(cb_r);
        copy_tile(cb_r, 0, 1);
        copy_from(cb_w);
        copy_tile(cb_w, wofs + j, 2);
        sfpu_mul_bcast_col_init();
        sfpu_mul_bcast_col(0, 1);
        sfpu_mul_bcast_row_init();
        sfpu_mul_bcast_row(0, 2);
        tile_regs_commit();
        tile_regs_wait();
        if (j < 2) {
            pack_tile<true>(0, cb_xn, j);
        } else {
            pack_tile<true>(0, cb_out, j);
        }
        tile_regs_release();
    }
    cxn.push_back(2);
    cr.pop_front(1);
    cx.pop_front(Ht);
    // RoPE on tiles 0 / 1
    cxn.wait_front(2);
    ccs.wait_front(4);  // cos0 cos1 sin0 sin1
    for (uint32_t t = 0; t < 2; t++) {
        tile_regs_acquire();
        copy_from(cb_xn);
        copy_tile(cb_xn, t, 0);
        copy_tile(cb_xn, 1 - t, 1);
        copy_from(cb_cs);
        copy_tile(cb_cs, t, 2);      // cos_t
        copy_tile(cb_cs, 2 + t, 3);  // sin_t
        sfpu_mul_bcast_row_init();
        sfpu_mul_bcast_row(0, 2);
        sfpu_mul_bcast_row(1, 3);
        if (t == 0) {
            sub_binary_tile_init();
            sub_binary_tile(0, 1, 0);
        } else {
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_out, t);
        tile_regs_release();
    }
    co.push_back(Ht);
    cxn.pop_front(2);
    ccs.pop_front(4);
}

template <uint32_t Ht>
void gate() {
    CircularBuffer cx(cb_x), co(cb_out);
    cx.wait_front(Ht);
    co.reserve_back(Ht);
    pack_reconfig_data_format(cb_out);
    copy_from(cb_x);
    sigmoid_tile_init();
    for (uint32_t j = 0; j < Ht; j++) {
        tile_regs_acquire();
        copy_tile(cb_x, j, 0);
        sigmoid_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
    }
    co.push_back(Ht);
    cx.pop_front(Ht);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(1);
    constexpr uint32_t inv_d_bits = get_compile_time_arg_val(2);
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    if (count == 0) {
        return;
    }
    compute_kernel_hw_startup(cb_x, cb_x, cb_out);
    CircularBuffer w(cb_w), ones(cb_ones);
    w.wait_front(2 * Ht);
    ones.wait_front(1);
    for (uint32_t i = 0; i < count; i++) {
        const uint32_t u = start + i;
        const uint32_t part = u % 3;
        if (part == 2) {
            gate<Ht>();
        } else {
            norm_rope<Ht>(part * Ht, eps_bits, inv_d_bits);
        }
    }
    w.pop_front(2 * Ht);
    ones.pop_front(1);
}
