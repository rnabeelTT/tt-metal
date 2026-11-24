// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fused decode post-attention section (tt/post.py), compute, on the live rows (VectorMode::R, ROWS <= 16):
//   y     = bf16(conv(att))                     (the attention-output short conv; states rotated)
//   resid = x + y   (ADD) | y                   (fp32)
//   r     = rsqrt(mean(resid^2) + eps), lg = r (resid . q)      (round 1: exact squares via hi/lo, TF32 dot)
//   w     = softmax([block logits..., lg])     (the block logits computed when the blocks were made)
//   mi    = sum_b w_b block_b + w_cur r resid
//   NORM: out = bf16(mi rsqrt(mean(mi^2) + eps) nw)           (round 2)   else out = mi (fp32)
// DST: full sync, 8 fp32 tiles.

#include <cstdint>

#include "api/compute/binary_max_min.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "tools/profiler/kernel_profiler.hpp"

#ifndef ROWS
#define ROWS 16
#endif
constexpr int RIT = 2 * ((ROWS + 3) / 4);

#ifdef TRISC_MATH
namespace ckernel::sfpu {
template <int OP, int ITERATIONS>
inline void calculate_post_bin(const uint32_t a, const uint32_t b, const uint32_t o) {
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
template <uint32_t MASK, int ITERATIONS>
inline void calculate_post_split(const uint32_t x, const uint32_t hi, const uint32_t lo) {
    constexpr uint32_t tile = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[x * tile];
        sfpi::vFloat h = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(v) & MASK);
        sfpi::dst_reg[hi * tile] = h;
        sfpi::dst_reg[lo * tile] = v - h;
        sfpi::dst_reg++;
    }
}
// round fp32 -> bf16 (nearest even), kept in fp32
template <int ITERATIONS>
inline void calculate_post_rbf16() {  // (the unary call points dst_reg at the tile)
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vUInt u = sfpi::as<sfpi::vUInt>(v);
        sfpi::vUInt lsb = (u >> 16) & 1;
        u = (u + 0x7FFF + lsb) & 0xFFFF0000;
        sfpi::dst_reg[0] = sfpi::as<sfpi::vFloat>(u);
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

namespace {
template <int OP>
ALWI void bin(uint32_t a, uint32_t b, uint32_t o) {
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_post_bin, (OP, RIT), a, b, o, VectorMode::R)));
}
ALWI void add(uint32_t a, uint32_t b, uint32_t o) { bin<0>(a, b, o); }
ALWI void sub(uint32_t a, uint32_t b, uint32_t o) { bin<1>(a, b, o); }
ALWI void mul(uint32_t a, uint32_t b, uint32_t o) { bin<2>(a, b, o); }
ALWI void sinit() { MATH((SFPU_BINARY_INIT(unused))); }
ALWI void split(uint32_t x, uint32_t hi, uint32_t lo) {
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_post_split, (0xFFFFE000u, RIT), x, hi, lo,
                           VectorMode::R)));
}
ALWI void rbf16(uint32_t x) {
    MATH((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_post_rbf16, (RIT), x, VectorMode::R)));
}
ALWI void drain() {
#ifdef TRISC_MATH
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
#endif
}
// The copy / pack configuration is redone only when the source / output kind changes (every copy source is an fp32
// unpack-to-dest CB but att / x (bf16), every output an fp32 CB but the bf16 out / router input; all of one kind
// share the format and tile shape), and a copy after a matmul re-inits.
constexpr uint32_t CB_ATT = get_compile_time_arg_val(0), CB_X = get_compile_time_arg_val(1);
constexpr uint32_t CB_OUT = get_compile_time_arg_val(22), OUT_BF = get_compile_time_arg_val(27);
uint32_t cp_kind = 2, pk_kind = 2;
ALWI void cp_prep(uint32_t cb) {
    drain();
    const uint32_t k = cb == CB_ATT || cb == CB_X;
    if (k != cp_kind) {
        reconfig_full_operand_srca(cb);  // (token-row CBs: TH x 32 tiles; matmul in1: 32 x 32)
        copy_init(cb);
        cp_kind = k;
    }
}
ALWI void cp(uint32_t cb, uint32_t i, uint32_t d) {
    drain();
    const uint32_t k = cb == CB_ATT || cb == CB_X;
    if (k != cp_kind) {
        reconfig_full_operand_srca(cb);  // (token-row CBs: TH x 32 tiles; matmul in1: 32 x 32)
        copy_init(cb);
        cp_kind = k;
    }
    copy_tile(cb, i, d);
}
ALWI void mm_begin(uint32_t in0, uint32_t in1) {
    drain();
    reconfig_full_operand(in1, in0);  // srcA = in1 (32 x 32), srcB = in0 (TH x 32: 4 FPU passes at TH = 8)
    matmul_init(in0, in1, 0);
    cp_kind = 2;
}
ALWI void pack_cfg(uint32_t cb) {
#if POST_RT
    const uint32_t k = (OUT_BF && cb == CB_OUT) || cb == POST_CB_XB;
#else
    const uint32_t k = OUT_BF && cb == CB_OUT;
#endif
    if (k != pk_kind) {
        pack_reconfig_data_format(cb);
        pk_kind = k;
    }
}
ALWI void pack_to(uint32_t d, uint32_t cb) {
    pack_cfg(cb);
    pack_tile(d, cb);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_att = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x = get_compile_time_arg_val(1);
    constexpr uint32_t cb_s = get_compile_time_arg_val(2);
    constexpr uint32_t cb_w = get_compile_time_arg_val(3);
    constexpr uint32_t cb_q = get_compile_time_arg_val(4);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(5);
    constexpr uint32_t cb_d = get_compile_time_arg_val(6);
    constexpr uint32_t cb_blk = get_compile_time_arg_val(7);
    constexpr uint32_t cb_nw = get_compile_time_arg_val(8);
    constexpr uint32_t cb_y16 = get_compile_time_arg_val(9);
    constexpr uint32_t cb_so = get_compile_time_arg_val(10);
    constexpr uint32_t cb_res = get_compile_time_arg_val(11);   // resid, kept (copies)
    constexpr uint32_t cb_rmm = get_compile_time_arg_val(12);   // resid, matmul operand
    constexpr uint32_t cb_rout = get_compile_time_arg_val(13);  // resid -> writer
    constexpr uint32_t cb_sq = get_compile_time_arg_val(14);
    constexpr uint32_t cb_part = get_compile_time_arg_val(15);
    constexpr uint32_t cb_gath = get_compile_time_arg_val(16);
    constexpr uint32_t cb_red = get_compile_time_arg_val(17);
    constexpr uint32_t cb_wt = get_compile_time_arg_val(18);
    constexpr uint32_t cb_mi = get_compile_time_arg_val(19);
    constexpr uint32_t cb_part2 = get_compile_time_arg_val(20);
    constexpr uint32_t cb_gath2 = get_compile_time_arg_val(21);
    constexpr uint32_t cb_out = get_compile_time_arg_val(22);
    constexpr uint32_t PER = get_compile_time_arg_val(23);
    constexpr uint32_t NB = get_compile_time_arg_val(24);
    constexpr uint32_t C = get_compile_time_arg_val(25);
    constexpr uint32_t ADD = get_compile_time_arg_val(26);
    constexpr uint32_t NORM = get_compile_time_arg_val(27);
    constexpr uint32_t inv_h = get_compile_time_arg_val(28);
    constexpr uint32_t eps = get_compile_time_arg_val(29);
    constexpr uint32_t cb_sel = get_compile_time_arg_val(30);  // row-select matrices (bf16 0 / 1), NQ tiles
    constexpr uint32_t NQ = get_compile_time_arg_val(31);      // tiles of gathered rows per part (C rows)
    constexpr uint32_t cb_tout = get_compile_time_arg_val(32);  // root: totals -> writer
    constexpr uint32_t cb_tot = get_compile_time_arg_val(33);   // round 1 totals (received)
    constexpr uint32_t cb_tot2 = get_compile_time_arg_val(34);  // round 2 totals (received)
    const bool root = get_arg_val<uint32_t>(0) < ROWS;  // core t: the root of row t
    constexpr uint32_t n = NB + 1;
    static_assert(n <= 6, "DST: max + n weights + sum");

    // matmuls: in0 -> srcB, in1 -> srcA (each CB's tile shape: the token-row CBs hold TH x 32 tiles, POST_TH)
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_sq, cb_ones, cb_so);

    {
    DeviceZoneScopedN("P_C_CONV");
    // ---- conv + residual, per tile
    cb_reserve_back(cb_res, PER);
    for (uint32_t i = 0; i < PER; ++i) {
        cb_wait_front(cb_att, 1);
        cb_wait_front(cb_s, 3);
        cb_wait_front(cb_w, 4);
        tile_regs_acquire();
        // DST: 0 x, 1 wc, 2 s0, 3 w0, 4 s1, 5 w1, 6 s2, 7 w2 (one unpack init per format)
        cp(cb_att, 0, 0);
        cp(cb_w, 3, 1);
        copy_tile(cb_s, 0, 2);
        copy_tile(cb_w, 0, 3);
        copy_tile(cb_s, 1, 4);
        copy_tile(cb_w, 1, 5);
        copy_tile(cb_s, 2, 6);
        copy_tile(cb_w, 2, 7);
        sinit();
        mul(0, 1, 1);  // y = x wc
        mul(2, 3, 3);
        add(1, 3, 1);
        mul(4, 5, 5);
        add(1, 5, 1);
        mul(6, 7, 7);
        add(1, 7, 1);
        rbf16(1);  // y = bf16(y) (the separate conv op returned bf16)
        if constexpr (ADD) {  // resid = x + y: x is the bf16 layer input (DST 2 is free now)
            cb_wait_front(cb_x, 1);
            cp(cb_x, 0, 2);
            sinit();
            add(1, 2, 1);
        }
        tile_regs_commit();
        cb_reserve_back(cb_so, 3);
        cb_reserve_back(cb_rout, 1);
        tile_regs_wait();
        pack_to(4, cb_so);  // rotated: s0 <- s1, s1 <- s2, s2 <- x
        pack_tile(6, cb_so);
        pack_tile(0, cb_so);
        pack_tile(1, cb_res);
        pack_tile(1, cb_rout);
        tile_regs_release();
        cb_push_back(cb_so, 3);
        cb_push_back(cb_rout, 1);
        cb_pop_front(cb_att, 1);
        cb_pop_front(cb_s, 3);
        cb_pop_front(cb_w, 4);
        if constexpr (ADD) {
            cb_pop_front(cb_x, 1);
        }
    }
    cb_push_back(cb_res, PER);
    cb_wait_front(cb_res, PER);

    }
    {
    DeviceZoneScopedN("P_C_SQ");
    // ---- round 1 partials: sum resid^2 (hi + lo, exact), resid . q
    cb_wait_front(cb_ones, 1);
    cb_wait_front(cb_q, PER);
    cb_reserve_back(cb_sq, 4 * PER);
    for (uint32_t i = 0; i < PER; ++i) {  // [resid^2 hi, lo, resid q hi, lo] per tile (fp32 products, exact sums)
        tile_regs_acquire();
        cp(cb_res, i, 0);
        copy_tile(cb_q, i, 1);
        sinit();
        mul(0, 1, 1);
        mul(0, 0, 0);
        split(0, 2, 3);
        split(1, 4, 5);
        tile_regs_commit();
        tile_regs_wait();
        pack_to(2, cb_sq);
        pack_tile(3, cb_sq);
        pack_tile(4, cb_sq);
        pack_tile(5, cb_sq);
        tile_regs_release();
    }
    cb_push_back(cb_sq, 4 * PER);
    cb_wait_front(cb_sq, 4 * PER);
    cb_reserve_back(cb_part, 4);
    tile_regs_acquire();
    mm_begin(cb_sq, cb_ones);
    for (uint32_t i = 0; i < PER; ++i) {
        matmul_tiles(cb_sq, cb_ones, 4 * i, 0, 0);
        matmul_tiles(cb_sq, cb_ones, 4 * i + 1, 0, 0);
        matmul_tiles(cb_sq, cb_ones, 4 * i + 2, 0, 1);
        matmul_tiles(cb_sq, cb_ones, 4 * i + 3, 0, 1);
    }
    sinit();
    // hi / lo of both partials (exact sums of the C parts through TF32 matmuls with 0 / 1 matrices)
    split(0, 2, 3);
    split(1, 4, 5);
    tile_regs_commit();
    tile_regs_wait();
    pack_to(2, cb_part);
    pack_tile(3, cb_part);
    pack_tile(4, cb_part);
    pack_tile(5, cb_part);
    tile_regs_release();
    cb_push_back(cb_part, 4);
    cb_pop_front(cb_sq, 4 * PER);
    cb_pop_front(cb_q, PER);

    }
    {
    DeviceZoneScopedN("P_C_R1");
    // ---- reduce round 1 -> r, lg
    if (root) {  // [sq hi | sq lo | dot hi | dot lo] x NQ tiles, row c = core c's row (this root's) -> totals row 0
        cb_wait_front(cb_sel, NQ);
        tile_regs_acquire();
        mm_begin(cb_sel, cb_gath);  // (configured before the gather arrives)
        cb_wait_front(cb_gath, 4 * NQ);
        for (uint32_t k = 0; k < NQ; ++k) {
            matmul_tiles(cb_sel, cb_gath, k, k, 0);
            matmul_tiles(cb_sel, cb_gath, k, NQ + k, 0);
            matmul_tiles(cb_sel, cb_gath, k, 2 * NQ + k, 1);
            matmul_tiles(cb_sel, cb_gath, k, 3 * NQ + k, 1);
        }
        tile_regs_commit();
        cb_reserve_back(cb_tout, 2);
        tile_regs_wait();
        pack_to(0, cb_tout);
        pack_tile(1, cb_tout);
        tile_regs_release();
        cb_push_back(cb_tout, 2);
        cb_pop_front(cb_gath, 4 * NQ);
    }
    tile_regs_acquire();
    cp_prep(cb_tot);  // (configured before the totals arrive)
    cb_wait_front(cb_tot, 2);
    cp(cb_tot, 0, 0);
    cp(cb_tot, 1, 1);
    drain();
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_h);
    add_unary_tile(0, eps);
    rsqrt_tile_init<false>();
    rsqrt_tile<false>(0);
    sinit();
    mul(1, 0, 1);  // lg = r dot
    tile_regs_commit();
    cb_reserve_back(cb_red, 2);
    tile_regs_wait();
    pack_to(0, cb_red);
    pack_tile(1, cb_red);
    tile_regs_release();
    cb_push_back(cb_red, 2);
    cb_pop_front(cb_tot, 2);
    cb_wait_front(cb_red, 2);

    }
    {
    DeviceZoneScopedN("P_C_SOFT");
    // ---- softmax over [block logits..., lg]: DST 0 = max, 1..n = e_i -> w_i, 7 = sum
    cb_wait_front(cb_d, NB);
    tile_regs_acquire();
#if SOFT_BATCH  // (each SFPU op over all sources: one init per op kind)
    cp(cb_red, 1, 0);
    for (uint32_t i = 0; i < n; ++i) {
        if (i < NB) {
            cp(cb_d, i, 1 + i);
        } else {
            cp(cb_red, 1, 1 + i);
        }
    }
    if constexpr (NB > 0) {
        drain();
        binary_max_tile_init();
        for (uint32_t b = 0; b < NB; ++b) {
            binary_max_tile(0, 1 + b, 0, VectorMode::R);
        }
    }
    drain();
    sinit();
    for (uint32_t i = 0; i < n; ++i) {
        sub(1 + i, 0, 1 + i);
    }
    drain();
    exp_tile_init();
    for (uint32_t i = 0; i < n; ++i) {
        exp_tile<false, false, InputClamping::ClampToNegative, RIT>(1 + i, VectorMode::R);
    }
#else
    cp(cb_red, 1, 0);
    for (uint32_t b = 0; b < NB; ++b) {
        cp(cb_d, b, 1);
        binary_max_tile_init();
        binary_max_tile(0, 1, 0, VectorMode::R);
    }
    for (uint32_t i = 0; i < n; ++i) {
        if (i < NB) {
            cp(cb_d, i, 1 + i);
        } else {
            cp(cb_red, 1, 1 + i);
        }
        sinit();
        sub(1 + i, 0, 1 + i);
        exp_tile_init();
        exp_tile<false, false, InputClamping::ClampToNegative, RIT>(1 + i, VectorMode::R);
    }
#endif
    if constexpr (n > 1) {  // (one source: w = exp(0) = 1)
        drain();
        sinit();
        add(1, 2, 7);
        for (uint32_t i = 2; i < n; ++i) {
            add(7, 1 + i, 7);
        }
        drain();
        recip_tile_init();
        recip_tile(7, VectorMode::R);
        sinit();
        for (uint32_t i = 0; i < n; ++i) {
            mul(1 + i, 7, 1 + i);
        }
    }
    cp(cb_red, 0, 7);  // the current source's weight times r (its source is resid, normed by r)
    sinit();
    mul(n, 7, n);
    tile_regs_commit();
    cb_reserve_back(cb_wt, n);
    tile_regs_wait();
    pack_cfg(cb_wt);
    for (uint32_t i = 0; i < n; ++i) {
        pack_tile(1 + i, cb_wt);
    }
    tile_regs_release();
    cb_push_back(cb_wt, n);
    cb_wait_front(cb_wt, n);
    cb_pop_front(cb_d, NB);

    }
    {
    DeviceZoneScopedN("P_C_MIX");
    // ---- mi = sum_b w_b block_b + w_cur r resid
    cb_wait_front(cb_blk, NB * PER);
    if constexpr (NORM) {
        cb_reserve_back(cb_mi, PER);
    }
    for (uint32_t i = 0; i < PER; ++i) {
        tile_regs_acquire();
        // DST: 0 acc, 1 resid, 2 w_cur r, then pairs (block, weight) at 3 + 2 b, 4 + 2 b (NB <= 2 at once)
        cp(cb_res, i, 1);
        copy_tile(cb_wt, NB, 2);
        sinit();
        mul(1, 2, 0);
        for (uint32_t b = 0; b < NB; ++b) {
            const uint32_t d = 3 + 2 * (b % 2);
            cp(cb_blk, b * PER + i, d);
            copy_tile(cb_wt, b, d + 1);
            sinit();
            mul(d, d + 1, d);
            add(0, d, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        if constexpr (NORM) {
            pack_to(0, cb_mi);
        } else {
            cb_reserve_back(cb_out, 1);
            pack_to(0, cb_out);
            cb_push_back(cb_out, 1);
        }
        tile_regs_release();
    }
    cb_pop_front(cb_blk, NB * PER);
    cb_pop_front(cb_res, PER);
    cb_pop_front(cb_wt, n);

    }
    {
    DeviceZoneScopedN("P_C_NORM");
    if constexpr (NORM) {
        cb_push_back(cb_mi, PER);
        cb_wait_front(cb_mi, PER);
        // round 2 partial: sum mi^2
        cb_reserve_back(cb_sq, 2 * PER);
        for (uint32_t i = 0; i < PER; ++i) {
            tile_regs_acquire();
            cp(cb_mi, i, 0);
            sinit();
            mul(0, 0, 0);
            split(0, 0, 1);
            tile_regs_commit();
            tile_regs_wait();
            pack_to(0, cb_sq);
            pack_tile(1, cb_sq);
            tile_regs_release();
        }
        cb_push_back(cb_sq, 2 * PER);
        cb_wait_front(cb_sq, 2 * PER);
        cb_reserve_back(cb_part2, 2);
        tile_regs_acquire();
        mm_begin(cb_sq, cb_ones);
        for (uint32_t i = 0; i < 2 * PER; ++i) {
            matmul_tiles(cb_sq, cb_ones, i, 0, 0);
        }
        sinit();
        split(0, 2, 3);
        tile_regs_commit();
        tile_regs_wait();
        pack_to(2, cb_part2);
        pack_tile(3, cb_part2);
        tile_regs_release();
        cb_push_back(cb_part2, 2);
        cb_pop_front(cb_sq, 2 * PER);
        // reduce -> rn (into cb_red slot 0 again: r is no longer needed)
        if (root) {
            tile_regs_acquire();
            mm_begin(cb_sel, cb_gath2);  // (configured before the gather arrives)
            cb_wait_front(cb_gath2, 2 * NQ);
            for (uint32_t k = 0; k < NQ; ++k) {
                matmul_tiles(cb_sel, cb_gath2, k, k, 0);
                matmul_tiles(cb_sel, cb_gath2, k, NQ + k, 0);
            }
            tile_regs_commit();
            cb_reserve_back(cb_tout, 1);
            tile_regs_wait();
            pack_to(0, cb_tout);
            tile_regs_release();
            cb_push_back(cb_tout, 1);
            cb_pop_front(cb_gath2, 2 * NQ);
        }
        tile_regs_acquire();
        cp_prep(cb_tot2);  // (configured before the totals arrive)
        cb_wait_front(cb_tot2, 1);
        cp(cb_tot2, 0, 0);
        drain();
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv_h);
        add_unary_tile(0, eps);
        rsqrt_tile_init<false>();
        rsqrt_tile<false>(0);
        tile_regs_commit();
        cb_pop_front(cb_red, 2);
        cb_reserve_back(cb_red, 1);
        tile_regs_wait();
        pack_to(0, cb_red);
        tile_regs_release();
        cb_push_back(cb_red, 1);
        cb_pop_front(cb_tot2, 1);
        cb_wait_front(cb_red, 1);
        cb_wait_front(cb_nw, PER);
        for (uint32_t i = 0; i < PER; ++i) {
            tile_regs_acquire();
            cp(cb_mi, i, 0);
            cp(cb_red, 0, 1);
            sinit();
            mul(0, 1, 0);
            cp(cb_nw, i, 1);
            sinit();
            mul(0, 1, 0);
            tile_regs_commit();
            cb_reserve_back(cb_out, 1);
            tile_regs_wait();
            pack_to(0, cb_out);
#if POST_RT
            pack_tile(0, POST_CB_XB);
#endif
            tile_regs_release();
            cb_push_back(cb_out, 1);
        }
        cb_pop_front(cb_red, 1);
        cb_pop_front(cb_mi, PER);
        cb_pop_front(cb_nw, PER);
#if POST_RT
        {  // router round: out (bf16) @ [W hi | W lo] rows of this core's tiles -> ET tiles -> hi / lo parts
            constexpr uint32_t ET = POST_RT, cb_xb = POST_CB_XB, cb_rw = POST_CB_RW, cb_p3 = POST_CB_P3;
            cb_push_back(cb_xb, PER);
            cb_wait_front(cb_xb, PER);
            cb_wait_front(cb_rw, 2 * PER * ET);
            tile_regs_acquire();
            mm_begin(cb_xb, cb_rw);
            for (uint32_t e = 0; e < ET; ++e) {
                for (uint32_t half = 0; half < 2; ++half) {
                    for (uint32_t i = 0; i < PER; ++i) {
                        matmul_tiles(cb_xb, cb_rw, i, (half * PER + i) * ET + e, e);
                    }
                }
            }
            sinit();
            for (uint32_t e = 0; e < ET; ++e) {
                split(e, e, 4 + e);
            }
            tile_regs_commit();
            cb_reserve_back(cb_p3, 2 * ET);
            tile_regs_wait();
            pack_cfg(cb_p3);
            for (uint32_t e = 0; e < ET; ++e) {
                pack_tile(e, cb_p3);
                pack_tile(4 + e, cb_p3);
            }
            tile_regs_release();
            cb_push_back(cb_p3, 2 * ET);
            cb_pop_front(cb_xb, PER);
            cb_pop_front(cb_rw, 2 * PER * ET);
#if POST_RSPLIT  // (core t ET + e reduces logit tile e of row t)
            const uint32_t ci = get_arg_val<uint32_t>(0);
            if (ci < ROWS * ET) {
                constexpr uint32_t cb_g3 = POST_CB_G3, cb_t3 = POST_CB_T3;
                const uint32_t e = ci % ET;
                cb_wait_front(cb_sel, NQ);
                tile_regs_acquire();
                mm_begin(cb_sel, cb_g3);  // (configured before the gather arrives)
                cb_wait_front(cb_g3, 2 * ET * NQ);
                for (uint32_t k = 0; k < NQ; ++k) {
                    matmul_tiles(cb_sel, cb_g3, k, (2 * e) * NQ + k, 0);
                    matmul_tiles(cb_sel, cb_g3, k, (2 * e + 1) * NQ + k, 0);
                }
                tile_regs_commit();
                cb_reserve_back(cb_t3, 1);
                tile_regs_wait();
                pack_cfg(cb_t3);
                pack_tile(0, cb_t3);
                tile_regs_release();
                cb_push_back(cb_t3, 1);
                cb_pop_front(cb_g3, 2 * ET * NQ);
                if (!root) {
                    cb_pop_front(cb_sel, NQ);
                }
            }
            if (false) {
#else
            if (root) {
#endif
                constexpr uint32_t cb_g3 = POST_CB_G3, cb_t3 = POST_CB_T3;
                tile_regs_acquire();
                mm_begin(cb_sel, cb_g3);  // (configured before the gather arrives)
                cb_wait_front(cb_g3, 2 * ET * NQ);
                for (uint32_t e = 0; e < ET; ++e) {
                    for (uint32_t k = 0; k < NQ; ++k) {
                        matmul_tiles(cb_sel, cb_g3, k, (2 * e) * NQ + k, e);
                        matmul_tiles(cb_sel, cb_g3, k, (2 * e + 1) * NQ + k, e);
                    }
                }
                tile_regs_commit();
                cb_reserve_back(cb_t3, ET);
                tile_regs_wait();
                pack_cfg(cb_t3);
                for (uint32_t e = 0; e < ET; ++e) {
                    pack_tile(e, cb_t3);
                }
                tile_regs_release();
                cb_push_back(cb_t3, ET);
                cb_pop_front(cb_g3, 2 * ET * NQ);
            }
        }
#endif
    } else {
        cb_pop_front(cb_red, 2);
    }
    }
    cb_pop_front(cb_ones, 1);
    if (root) {
        cb_pop_front(cb_sel, NQ);
    }
}
