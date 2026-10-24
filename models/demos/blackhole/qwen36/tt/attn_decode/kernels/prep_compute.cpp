// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Attention decode prep, compute. Item u = 3 b + part; the heads of the item are the rows of Ht bf16 tiles (cb_x):
//   part 0 / 1 (q / k): r = 1 / sqrt(mean_d x^2 + eps) per row, in every column (squares, then a matmul with the
//     all-ones tile summing each row over the Ht tiles), x (1 + w) (FPU row broadcast) r (FPU, DST reuse), then the HF rotate-half RoPE on tiles 0 / 1 (the first rope_dim = 64 dims): y0 = x0 cos0 - x1 sin0,
//     y1 = x1 cos1 + x0 sin1 (FPU row broadcasts, SFPU add / sub);
//   part 2 (gate): sigmoid on rows 0..15 (the heads; NH <= 16).
// Only the head rows carry data; the other rows hold whatever the CB had and are never read downstream.
// Results bf16 to cb_out in tile order.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"

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
    DeviceZoneScopedN("normrope");
    // squares (FP32), 4 tiles per DST acquire
    csq.reserve_back(Ht);
    pack_reconfig_data_format(cb_sq);
    reconfig_data_format(cb_x, cb_x);
    mul_init(cb_x, cb_x);
    for (uint32_t j0 = 0; j0 < Ht; j0 += 4) {
        tile_regs_acquire();
        for (uint32_t j = j0; j < j0 + 4 && j < Ht; j++) {
            mul_tiles(cb_x, cb_x, j, j, j - j0);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = j0; j < j0 + 4 && j < Ht; j++) {
            pack_tile(j - j0, cb_sq);
        }
        tile_regs_release();
    }
    csq.push_back(Ht);
    // r = rsqrt(sum / D + eps) per row, every column
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
    // x (1 + w) (row broadcast) r (DST reuse): tiles 0 / 1 -> cb_xn (FP32, for the RoPE), the rest -> cb_out (bf16)
    co.reserve_back(Ht);
    cxn.reserve_back(2);
    for (uint32_t j = 0; j < Ht; j++) {
        if (j == 0) {
            pack_reconfig_data_format(cb_xn);
        } else if (j == 2) {
            pack_reconfig_data_format(cb_out);
        }
        tile_regs_acquire();
        reconfig_data_format(cb_x, cb_w);
        mul_bcast_rows_init(cb_x, cb_w);
        mul_tiles_bcast_rows(cb_x, cb_w, j, wofs + j, 0);
        reconfig_data_format_srcb(cb_r);
        mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_r);
        mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_r, 0, 0);
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
    // RoPE on tiles 0 / 1: products on the FPU (row 0 of the user's cos / sin tile broadcast), then SFPU sub / add
    cxn.wait_front(2);
    ccs.wait_front(4);  // cos0 cos1 sin0 sin1
    reconfig_data_format(cb_xn, cb_cs);
    for (uint32_t t = 0; t < 2; t++) {
        tile_regs_acquire();
        mul_bcast_rows_init(cb_xn, cb_cs);
        mul_tiles_bcast_rows(cb_xn, cb_cs, t, t, 0);          // x_t cos_t
        mul_tiles_bcast_rows(cb_xn, cb_cs, 1 - t, 2 + t, 1);  // x_{1-t} sin_t
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
    DeviceZoneScopedN("gate");
    co.reserve_back(Ht);
    pack_reconfig_data_format(cb_out);
    copy_from(cb_x);
    sigmoid_tile_init();
    for (uint32_t j0 = 0; j0 < Ht; j0 += 4) {
        tile_regs_acquire();
        for (uint32_t j = j0; j < j0 + 4 && j < Ht; j++) {
            copy_tile(cb_x, j, j - j0);
            sigmoid_tile<VectorMode::R>(j - j0);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = j0; j < j0 + 4 && j < Ht; j++) {
            pack_tile(j - j0, cb_out);
        }
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
