// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Block attention residual mix (tt/mixw.py), compute (fp32 DST, SFPU): per query q,
//   w_i = softmax_i(l_iq)                    (elementwise over the n logit tiles: every column holds the logit)
//   out_q[t] = sum_i w_i * S_i[t]            (this core's H tiles t)
// DST (full sync, 8 fp32 tiles): softmax: 0 = max, 1..n = e_i / w_i, 7 = sum; weighted sum: 0 = acc, 1 / 2 = operands.

#include <cstdint>

#include "api/compute/binary_max_min.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

// MIX_ROWS (decode, <= 16 live rows): the SFPU work on the live rows only (faces 0 / 1, 2 ceil(T / 4) iterations per
// face: VectorMode::R) instead of all 32 rows of every tile; the dead rows of the outputs are don't-care
#ifndef MIX_ROWS
#define MIX_ROWS 32
#endif
constexpr bool ROWS = MIX_ROWS <= 16;
constexpr int RIT = 2 * ((MIX_ROWS + 3) / 4);

#ifdef TRISC_MATH
namespace ckernel::sfpu {
// DST[o] = DST[a] op DST[b] over the first ITERATIONS SFPU rows of the faces the vector mode selects (op 0 add, 1 sub,
// 2 mul)
template <int OP, int ITERATIONS>
inline void calculate_mix_bin(const uint32_t a, const uint32_t b, const uint32_t o) {
    constexpr uint32_t tile = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[a * tile];
        sfpi::vFloat y = sfpi::dst_reg[b * tile];
        if constexpr (OP == 0) {
            sfpi::dst_reg[o * tile] = x + y;
        } else if constexpr (OP == 1) {
            sfpi::dst_reg[o * tile] = x - y;
        } else {
            sfpi::dst_reg[o * tile] = x * y;
        }
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

namespace {
template <int OP>
ALWI void rbin(uint32_t a, uint32_t b, uint32_t o) {
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_mix_bin, (OP, RIT), a, b, o, VectorMode::R)));
}
ALWI void sfpu_drain() {
#ifdef TRISC_MATH
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
#endif
}
ALWI void bin_init() { MATH((SFPU_BINARY_INIT(unused))); }
ALWI void op_add(uint32_t a, uint32_t b, uint32_t o) {
    if constexpr (ROWS) { rbin<0>(a, b, o); } else { add_binary_tile(a, b, o); }
}
ALWI void op_sub(uint32_t a, uint32_t b, uint32_t o) {
    if constexpr (ROWS) { rbin<1>(a, b, o); } else { sub_binary_tile(a, b, o); }
}
ALWI void op_mul(uint32_t a, uint32_t b, uint32_t o) {
    if constexpr (ROWS) { rbin<2>(a, b, o); } else { mul_binary_tile(a, b, o); }
}
ALWI void init_add() { if constexpr (ROWS) { bin_init(); } else { add_binary_tile_init(); } }
ALWI void init_sub() { if constexpr (ROWS) { bin_init(); } else { sub_binary_tile_init(); } }
ALWI void init_mul() { if constexpr (ROWS) { bin_init(); } else { mul_binary_tile_init(); } }
constexpr VectorMode VM = ROWS ? VectorMode::R : VectorMode::RC;
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_lg = get_compile_time_arg_val(0);
    constexpr uint32_t cb_s = get_compile_time_arg_val(1);
    constexpr uint32_t cb_w = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t n = get_compile_time_arg_val(4);
    constexpr uint32_t nq = get_compile_time_arg_val(5);
    const uint32_t nt = get_arg_val<uint32_t>(0);
    const uint32_t RT = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_lg, cb_out);
    pack_reconfig_data_format(cb_w);
    for (uint32_t r = 0; r < RT; ++r) {
    cb_wait_front(cb_lg, n * nq);
    // ---- softmax weights over the n sources, every query (cb_w: query q's n weights at q * n ..)
    cb_reserve_back(cb_w, n * nq);
    for (uint32_t q = 0; q < nq; ++q) {
        tile_regs_acquire();
        copy_init(cb_lg);
        copy_tile(cb_lg, q, 0);
        for (uint32_t i = 1; i < n; ++i) {
            if constexpr (ROWS) { sfpu_drain(); }
            copy_init(cb_lg);
            copy_tile(cb_lg, i * nq + q, 1);
            binary_max_tile_init();
            binary_max_tile(0, 1, 0, VM);
        }
        for (uint32_t i = 0; i < n; ++i) {
            if constexpr (ROWS) { sfpu_drain(); }
            copy_init(cb_lg);
            copy_tile(cb_lg, i * nq + q, 1 + i);
            init_sub();
            op_sub(1 + i, 0, 1 + i);
            exp_tile_init();
            if constexpr (ROWS) {
                exp_tile<false, false, InputClamping::ClampToNegative, RIT>(1 + i, VectorMode::R);
            } else {
                exp_tile(1 + i);
            }
        }
        if constexpr (n == 1) {
            // a single source: w = exp(0) = 1 already
        } else {
            init_add();
            op_add(1, 2, 7);
            for (uint32_t i = 2; i < n; ++i) {
                op_add(7, 1 + i, 7);
            }
            recip_tile_init();
            recip_tile(7, VM);
            init_mul();
            for (uint32_t i = 0; i < n; ++i) {
                op_mul(1 + i, 7, 1 + i);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(1 + i, cb_w);  // (in order: q * n + i)
        }
        tile_regs_release();
    }
    cb_push_back(cb_w, n * nq);
    cb_pop_front(cb_lg, n * nq);
    cb_wait_front(cb_w, n * nq);
    // ---- weighted sums over this core's H tiles (the n source tiles of column t arrive together)
    copy_init(cb_s);  // (cb_s / cb_w: the same format and tile shape; the SFPU inits leave the copy config alone)
    for (uint32_t t = 0; t < nt; ++t) {
        cb_wait_front(cb_s, n);
        for (uint32_t q = 0; q < nq; ++q) {
            tile_regs_acquire();
                        copy_tile(cb_s, 0, 0);
                        copy_tile(cb_w, q * n, 1);
            init_mul();
            op_mul(0, 1, 0);
            for (uint32_t i = 1; i < n; ++i) {
                if constexpr (ROWS) { sfpu_drain(); }
                                copy_tile(cb_s, i, 1);
                                copy_tile(cb_w, q * n + i, 2);
                init_mul();
                op_mul(1, 2, 1);
                init_add();
                op_add(0, 1, 0);
            }
            tile_regs_commit();
            cb_reserve_back(cb_out, 1);
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, 1);
        }
        cb_pop_front(cb_s, n);
    }
    cb_pop_front(cb_w, n * nq);
    }
}
