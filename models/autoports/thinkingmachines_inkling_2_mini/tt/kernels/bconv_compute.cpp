// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Batched short conv step (tt/bconv.py), compute: per channel tile, in FP32 DST (unpack-to-dest FP32, SFPU math):
//   y = x * wc + s0 * w0 + s1 * w1 + s2 * w2        -> cb_y (bf16)
//   the rotated states: s1 -> cb_o0, s2 -> cb_o1, x (as fp32) -> cb_o2 (the writer stores them as s0, s1, s2)
// DST tiles: 0 = x, 1 / 2 = operands, 4 = y accumulator (full-sync DST: 8 FP32 tiles).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

// BCONV_ROWS (<= 16 live rows, the decode): the SFPU multiply / adds on those rows only (VectorMode::R, 2 ceil(B / 4)
// iterations per face); the padding rows of y are don't-care (the states are rotated as whole tiles)
#ifndef BCONV_ROWS
#define BCONV_ROWS 32
#endif
constexpr bool CROWS = BCONV_ROWS <= 16;
constexpr int CIT = 2 * ((BCONV_ROWS + 3) / 4);

#ifdef TRISC_MATH
namespace ckernel::sfpu {
template <int OP, int ITERATIONS>
inline void calculate_bconv_bin(const uint32_t a, const uint32_t b, const uint32_t o) {
    constexpr uint32_t tile = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[a * tile];
        sfpi::vFloat y = sfpi::dst_reg[b * tile];
        if constexpr (OP == 0) {
            sfpi::dst_reg[o * tile] = x + y;
        } else {
            sfpi::dst_reg[o * tile] = x * y;
        }
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

namespace {
ALWI void drain() {
#ifdef TRISC_MATH
    if constexpr (CROWS) {
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
    }
#endif
}
ALWI void c_mul(uint32_t a, uint32_t b, uint32_t o) {
    if constexpr (CROWS) {
        MATH((SFPU_BINARY_INIT(unused)));
        MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_bconv_bin, (2, CIT), a, b, o, VectorMode::R)));
    } else {
        mul_binary_tile_init();
        mul_binary_tile(a, b, o);
    }
}
ALWI void c_add(uint32_t a, uint32_t b, uint32_t o) {
    if constexpr (CROWS) {
        MATH((SFPU_BINARY_INIT(unused)));
        MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_bconv_bin, (0, CIT), a, b, o, VectorMode::R)));
    } else {
        add_binary_tile_init();
        add_binary_tile(a, b, o);
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_s = get_compile_time_arg_val(1);
    constexpr uint32_t cb_w = get_compile_time_arg_val(2);
    constexpr uint32_t cb_y = get_compile_time_arg_val(3);
    constexpr uint32_t cb_o0 = get_compile_time_arg_val(4);
    constexpr uint32_t cb_o1 = get_compile_time_arg_val(5);
    constexpr uint32_t cb_o2 = get_compile_time_arg_val(6);
    const uint32_t nt = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_s, cb_y);
#ifdef BCONV_DYN
    constexpr uint32_t cb_d = 8, cb_e = 9, cb_sc = 10;
    cb_wait_front(cb_d, 1);
#endif
    for (uint32_t t = 0; t < nt; ++t) {
#ifdef BCONV_DYN
        // tap j = d @ e_j (exact: one 1 per column) + sc_j (fp32, SFPU) -> cb_w, then the static path below
        cb_wait_front(cb_e, 4);
        cb_wait_front(cb_sc, 4);
        cb_reserve_back(cb_w, 4);
        tile_regs_acquire();
        reconfig_data_format(cb_e, cb_d);  // matmul: in1 -> srcA, in0 -> srcB
        matmul_init(cb_d, cb_e, 0);
        for (uint32_t j = 0; j < 4; ++j) {
            matmul_tiles(cb_d, cb_e, 0, j, j);
        }
        reconfig_data_format_srca(cb_sc);
        copy_init(cb_sc);
        for (uint32_t j = 0; j < 4; ++j) {
            copy_tile(cb_sc, j, 4 + j);
        }
        for (uint32_t j = 0; j < 4; ++j) {
            c_add(j, 4 + j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_w);
        for (uint32_t j = 0; j < 4; ++j) {
            pack_tile(j, cb_w);
        }
        tile_regs_release();
        cb_push_back(cb_w, 4);
        cb_pop_front(cb_e, 4);
        cb_pop_front(cb_sc, 4);
#endif
        cb_wait_front(cb_x, 1);
        cb_wait_front(cb_s, 3);
        cb_wait_front(cb_w, 4);
        tile_regs_acquire();
        // y = x * wc
        reconfig_data_format_srca(cb_x);
        copy_init(cb_x);
        copy_tile(cb_x, 0, 0);
        reconfig_data_format_srca(cb_w);
        copy_init(cb_w);
        copy_tile(cb_w, 3, 1);
        c_mul(0, 1, 4);
        // y += s_j * w_j
        for (uint32_t j = 0; j < 3; ++j) {
            drain();
            reconfig_data_format_srca(cb_s);
            copy_init(cb_s);
            copy_tile(cb_s, j, 1);
            reconfig_data_format_srca(cb_w);
            copy_init(cb_w);
            copy_tile(cb_w, j, 2);
            c_mul(1, 2, 1);
            c_add(4, 1, 4);
        }
        // the rotated states (exact FP32 copies through DST)
        drain();
        reconfig_data_format_srca(cb_s);
        copy_init(cb_s);
        copy_tile(cb_s, 1, 1);
        copy_tile(cb_s, 2, 2);
        tile_regs_commit();
        cb_reserve_back(cb_y, 1);
        cb_reserve_back(cb_o0, 1);
        cb_reserve_back(cb_o1, 1);
        cb_reserve_back(cb_o2, 1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_y);
        pack_tile(4, cb_y);
        pack_reconfig_data_format(cb_o0);
        pack_tile(1, cb_o0);
        pack_tile(2, cb_o1);
        pack_tile(0, cb_o2);
        tile_regs_release();
        cb_push_back(cb_y, 1);
        cb_push_back(cb_o0, 1);
        cb_push_back(cb_o1, 1);
        cb_push_back(cb_o2, 1);
        cb_pop_front(cb_x, 1);
        cb_pop_front(cb_s, 3);
        cb_pop_front(cb_w, 4);
    }
#ifdef BCONV_DYN
    cb_pop_front(cb_d, 1);
#endif
}
