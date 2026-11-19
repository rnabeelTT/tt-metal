// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fused hyper-connection (decode), compute. Core c owns hidden tiles [c J, c J + J) of the 4 streams (tile s * J + jj).
//
// POST (the previous sublayer's HC post): Q_s = wp @ sel_{4+s} (every column = stream s's post weight),
//   X'_{s,j} = bf16(X_{s,j} + o_j * Q_s)  (the product exact on the FPU, the add fp32 on the SFPU).
// PRE (the next sublayer's HC pre):
//   partial mix = sum_t X'_t @ fn_t (fn: this core's FP32 rows of fn^T);
//   partial S   = sum_t X'_t @ X'_t^T (its diagonal = this core's per-row sum of squares; bf16 products exact);
//   the two tiles go to every core (writer); then, in FP32 (unpack-to-dest copies + SFPU arithmetic):
//   M = sum mix, ss = ((sum S) * I) @ ones, r = rsqrt(ss / n + eps), w = aff * sigmoid(M r * scale + base) + hc_eps;
//   P_s = w @ sel_s; in_j = bf16(sum_s X'_{s,j} * P_s).
// The [32, 32] weight tile w (cols 0..hc-1 pre, hc..2hc-1 post) goes out for the next HC post (core 0 writes it).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

namespace {
constexpr uint32_t cb_x = 0, cb_o = 1, cb_wp = 2, cb_fn = 3, cb_mm = 4, cb_cv = 5, cb_q = 6, cb_xn = 7, cb_s1 = 8,
                   cb_p = 9, cb_slots = 10, cb_w = 11, cb_wm = 12, cb_pre = 13, cb_m = 14, cb_in = 16, cb_wpc = 17,
                   cb_wc = 18, cb_hi = 19, cb_lo = 20, cb_sdc = 21, cb_fnlo = 22;
constexpr uint32_t MM_ONES = 0, MM_SEL = 1, MM_EYE = 9;
// cb_s1 (S * I, a matmul operand: default unpack) and cb_m (M, FP32 unpack-to-dest copies) must be separate CBs
// cb_mm: 0 ones, 1 + m: selector of column m (row m all ones), 9 identity; cb_cv: 0 scale, 1 base, 2 affine rows, 3 identity
// (copy mode). FP32 operands of a matmul enter the FPU at tf32: split_hi_lo makes an FP32 tile W (in cb_mm-able cb_src
// and in copy-mode cb_copy) into hi = W @ I (W truncated to tf32, exactly) and lo = W - hi (exact on the SFPU), so
// hi @ B + lo @ B equals W @ B to ~fp32 for the selector / ones matmuls.

void split_hi_lo(uint32_t cb_src, uint32_t cb_copy) {
    cb_reserve_back(cb_hi, 1);
    pack_reconfig_data_format(cb_hi);
    reconfig_data_format(cb_mm, cb_src);
    matmul_init(cb_src, cb_mm, 0);
    tile_regs_acquire();
    matmul_tiles(cb_src, cb_mm, 0, MM_EYE, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_hi);
    tile_regs_release();
    cb_push_back(cb_hi, 1);
    cb_wait_front(cb_hi, 1);
    cb_reserve_back(cb_lo, 1);
    tile_regs_acquire();
    reconfig_data_format_srca(cb_copy);
    copy_init(cb_copy);
    copy_tile(cb_copy, 0, 0);
    reconfig_data_format_srca(cb_hi);
    copy_init(cb_hi);
    copy_tile(cb_hi, 0, 1);  // tf32 values: exact through srcA
    sub_binary_tile_init();
    sub_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_lo);
    pack_tile(0, cb_lo);
    tile_regs_release();
    cb_push_back(cb_lo, 1);
    cb_wait_front(cb_lo, 1);
}

// DST[d + i] = (hi + lo) @ mm[t0 + i], i < n (the caller has acquired DST and pops cb_hi / cb_lo afterwards)
void mm_hi_lo(uint32_t t0, uint32_t n, uint32_t d) {
    reconfig_data_format(cb_mm, cb_hi);
    matmul_init(cb_hi, cb_mm, 0);
    for (uint32_t i = 0; i < n; ++i) {
        matmul_tiles(cb_hi, cb_mm, 0, t0 + i, d + i);
        matmul_tiles(cb_lo, cb_mm, 0, t0 + i, d + i);
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t J = get_compile_time_arg_val(0);
    constexpr uint32_t POST = get_compile_time_arg_val(1);
    constexpr uint32_t PRE = get_compile_time_arg_val(2);
    constexpr uint32_t NC = get_compile_time_arg_val(3);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(4);
    constexpr uint32_t inv_n_bits = get_compile_time_arg_val(5);
    constexpr uint32_t heps_bits = get_compile_time_arg_val(6);
    constexpr uint32_t DBG = get_compile_time_arg_val(7);
    constexpr uint32_t FNLO = get_compile_time_arg_val(8);  // the lo part of fn (HY4_HCF_FNLO)  // 1: w := sum of mix partials, 2: w := ss, 3: w := sum S
    constexpr uint32_t NT = 4 * J;

    compute_kernel_hw_startup(cb_x, cb_x, cb_xn);
    cb_wait_front(cb_x, NT);
    cb_wait_front(cb_mm, 10);
    uint32_t src = cb_x;
    if constexpr (POST) {
        cb_wait_front(cb_wp, 1);
        cb_wait_front(cb_o, J);
        cb_wait_front(cb_wpc, 1);
        split_hi_lo(cb_wp, cb_wpc);
        cb_reserve_back(cb_q, 4);
        pack_reconfig_data_format(cb_q);
        tile_regs_acquire();
        mm_hi_lo(MM_SEL + 4, 4, 0);
        cb_pop_front(cb_hi, 1);
        cb_pop_front(cb_lo, 1);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t s = 0; s < 4; ++s) {
            pack_tile(s, cb_q);
        }
        tile_regs_release();
        cb_push_back(cb_q, 4);
        cb_wait_front(cb_q, 4);
        cb_reserve_back(cb_xn, NT);
        pack_reconfig_data_format(cb_xn);
        for (uint32_t s = 0; s < 4; ++s) {
            for (uint32_t j = 0; j < J; j += 2) {
                const uint32_t n = J - j < 2 ? J - j : 2;
                tile_regs_acquire();
                reconfig_data_format(cb_o, cb_q);
                mul_init(cb_o, cb_q, false);
                for (uint32_t i = 0; i < n; ++i) {
                    mul_tiles(cb_o, cb_q, j + i, s, 2 * i);
                }
                reconfig_data_format_srca(cb_x);
                copy_init(cb_x);
                for (uint32_t i = 0; i < n; ++i) {
                    copy_tile(cb_x, s * J + j + i, 2 * i + 1);
                }
                add_binary_tile_init();
                for (uint32_t i = 0; i < n; ++i) {
                    add_binary_tile(2 * i, 2 * i + 1, 2 * i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < n; ++i) {
                    pack_tile(2 * i, cb_xn);
                }
                tile_regs_release();
            }
        }
        cb_push_back(cb_xn, NT);
        cb_wait_front(cb_xn, NT);
        src = cb_xn;
        cb_pop_front(cb_q, 4);
        cb_pop_front(cb_o, J);
        cb_pop_front(cb_wp, 1);
        cb_pop_front(cb_wpc, 1);
    }
    if constexpr (PRE) {
        cb_wait_front(cb_fn, NT);
        cb_wait_front(cb_cv, 4);
        // this core's partials
        cb_reserve_back(cb_p, 2);
        pack_reconfig_data_format(cb_p);
        tile_regs_acquire();
        reconfig_data_format(cb_fn, src);
        matmul_init(src, cb_fn, 0);
        for (uint32_t t = 0; t < NT; ++t) {
            matmul_tiles(src, cb_fn, t, t, 0);
        }
        if constexpr (FNLO) {
            cb_wait_front(cb_fnlo, NT);
            reconfig_data_format(cb_fnlo, src);
            matmul_init(src, cb_fnlo, 0);
            for (uint32_t t = 0; t < NT; ++t) {
                matmul_tiles(src, cb_fnlo, t, t, 0);
            }
            cb_pop_front(cb_fnlo, NT);
        }
        reconfig_data_format(src, src);
        matmul_init(src, src, 1);  // in1 transposed: X'_t @ X'_t^T
        for (uint32_t t = 0; t < NT; ++t) {
            matmul_tiles(src, src, t, t, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_p);
        pack_tile(1, cb_p);
        tile_regs_release();
        cb_push_back(cb_p, 2);

        // every core's partials (FP32 exact sums)
        cb_wait_front(cb_slots, 2 * NC);
        cb_reserve_back(cb_s1, 1);
        cb_reserve_back(cb_sdc, 1);
        cb_reserve_back(cb_m, 1);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_slots);
        copy_init(cb_slots);
        copy_tile(cb_slots, 0, 0);
        copy_tile(cb_slots, 1, 1);
        for (uint32_t k = 1; k < NC; ++k) {
            reconfig_data_format_srca(cb_slots);
            copy_init(cb_slots);
            copy_tile(cb_slots, 2 * k, 2);
            copy_tile(cb_slots, 2 * k + 1, 3);
            add_binary_tile_init();
            add_binary_tile(0, 2, 0);
            add_binary_tile(1, 3, 1);
        }
        reconfig_data_format_srca(cb_cv);
        copy_init(cb_cv);
        copy_tile(cb_cv, 3, 2);  // identity
        mul_binary_tile_init();
        mul_binary_tile(1, 2, 1);  // the diagonal of S
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_s1);
        pack_tile(1, cb_s1);
        pack_tile(1, cb_sdc);
        pack_tile(0, cb_m);
        tile_regs_release();
        cb_push_back(cb_s1, 1);
        cb_push_back(cb_sdc, 1);
        cb_push_back(cb_m, 1);
        cb_pop_front(cb_slots, 2 * NC);
        cb_wait_front(cb_s1, 1);
        cb_wait_front(cb_sdc, 1);
        cb_wait_front(cb_m, 1);
        split_hi_lo(cb_s1, cb_sdc);

        cb_reserve_back(cb_w, 1);
        cb_reserve_back(cb_wm, 1);
        tile_regs_acquire();
        mm_hi_lo(MM_ONES, 1, 1);  // ss in every column
        cb_pop_front(cb_hi, 1);
        cb_pop_front(cb_lo, 1);
        reconfig_data_format_srca(cb_m);
        copy_init(cb_m);
        copy_tile(cb_m, 0, 0);  // M
        binop_with_scalar_tile_init();
        mul_unary_tile(1, inv_n_bits);
        add_unary_tile(1, eps_bits);
        rsqrt_tile_init();
        rsqrt_tile(1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);  // M r
        reconfig_data_format_srca(cb_cv);
        copy_init(cb_cv);
        copy_tile(cb_cv, 0, 1);  // scale
        copy_tile(cb_cv, 1, 2);  // base
        copy_tile(cb_cv, 2, 3);  // affine
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        add_binary_tile_init();
        add_binary_tile(0, 2, 0);
        sigmoid_tile_init();
        sigmoid_tile(0);
        mul_binary_tile_init();
        mul_binary_tile(0, 3, 0);
        binop_with_scalar_tile_init();
        add_unary_tile(0, heps_bits);
        if constexpr (DBG == 1) {
            reconfig_data_format_srca(cb_m);
            copy_init(cb_m);
            copy_tile(cb_m, 0, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_w);
        pack_tile(0, cb_w);
        pack_tile(0, cb_wm);
        pack_tile(0, cb_wc);
        tile_regs_release();
        cb_push_back(cb_w, 1);
        cb_push_back(cb_wm, 1);
        cb_push_back(cb_wc, 1);
        cb_pop_front(cb_s1, 1);
        cb_pop_front(cb_sdc, 1);
        cb_pop_front(cb_m, 1);
        cb_wait_front(cb_wm, 1);
        cb_wait_front(cb_wc, 1);

        // P_s = w @ sel_s (hi / lo: the pre weights to ~fp32)
        split_hi_lo(cb_wm, cb_wc);
        cb_reserve_back(cb_pre, 4);
        pack_reconfig_data_format(cb_pre);
        tile_regs_acquire();
        mm_hi_lo(MM_SEL, 4, 0);
        cb_pop_front(cb_hi, 1);
        cb_pop_front(cb_lo, 1);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t s = 0; s < 4; ++s) {
            pack_tile(s, cb_pre);
        }
        tile_regs_release();
        cb_push_back(cb_pre, 4);
        cb_wait_front(cb_pre, 4);

        // in_j = bf16(sum_s X'_{s,j} P_s)
        cb_reserve_back(cb_in, J);
        pack_reconfig_data_format(cb_in);
        for (uint32_t j = 0; j < J; ++j) {
            tile_regs_acquire();
            reconfig_data_format(src, cb_pre);
            mul_init(src, cb_pre, false);
            for (uint32_t s = 0; s < 4; ++s) {
                mul_tiles(src, cb_pre, s * J + j, s, s);
            }
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            add_binary_tile(2, 3, 2);
            add_binary_tile(0, 2, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_in);
            tile_regs_release();
        }
        cb_push_back(cb_in, J);
        cb_pop_front(cb_pre, 4);
        cb_pop_front(cb_wm, 1);
        cb_pop_front(cb_wc, 1);
        cb_pop_front(cb_fn, NT);
        cb_pop_front(cb_cv, 4);
    }
    cb_pop_front(cb_x, NT);
    cb_pop_front(cb_mm, 10);
}
