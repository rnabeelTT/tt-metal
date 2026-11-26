// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fused decode pre-attention section (tt/pre.py), compute, on the live rows (VectorMode::R, ROWS <= 16):
//   TAIL: x = bf16(resid + bf16(conv(mo)))           (the previous application's MoE-output conv and residual)
//   r = rsqrt(mean(x^2) + eps), lg_j = r (x . q_j)   (round 1; exact fp32 products, hi / lo sums)
//   h_j = sum_b w_bj block_b + w_cur_j r x            (w_.j = softmax over [block logits..., lg_j])
//   BOUNDARY: xn = r x (the new block)
//   NORM: out_j = bf16(h_j rsqrt(mean(h_j^2) + eps) nw_j)  (round 2)   else out_j = h_j (fp32: message gates follow)

#include <cstdint>

#include "api/compute/binary_max_min.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "glz.hpp"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

#ifndef ROWS
#define ROWS 16
#endif
constexpr int RIT = 2 * ((ROWS + 3) / 4);

#ifdef TRISC_MATH
namespace ckernel::sfpu {
template <int OP, int ITERATIONS>
inline void calculate_pre_bin(const uint32_t a, const uint32_t b, const uint32_t o) {
    constexpr uint32_t tile = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[a * tile];
        sfpi::vFloat y = sfpi::dst_reg[b * tile];
        if constexpr (OP == 0) {
            sfpi::dst_reg[o * tile] = x + y;
        } else if constexpr (OP == 1) {
            sfpi::dst_reg[o * tile] = x - y;
        } else if constexpr (OP == 2) {
            sfpi::dst_reg[o * tile] = x * y;
        } else {  // (a copy)
            sfpi::dst_reg[o * tile] = x;
        }
        sfpi::dst_reg++;
    }
}
template <uint32_t MASK, int ITERATIONS>
inline void calculate_pre_split(const uint32_t x, const uint32_t hi, const uint32_t lo) {
    constexpr uint32_t tile = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[x * tile];
        sfpi::vFloat h = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(v) & MASK);
        sfpi::dst_reg[hi * tile] = h;
        sfpi::dst_reg[lo * tile] = v - h;
        sfpi::dst_reg++;
    }
}
template <int ITERATIONS>
inline void calculate_pre_rbf16() {
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
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_pre_bin, (OP, RIT), a, b, o, VectorMode::R)));
}
ALWI void add(uint32_t a, uint32_t b, uint32_t o) { bin<0>(a, b, o); }
ALWI void sub(uint32_t a, uint32_t b, uint32_t o) { bin<1>(a, b, o); }
ALWI void mul(uint32_t a, uint32_t b, uint32_t o) { bin<2>(a, b, o); }
ALWI void mov(uint32_t a, uint32_t o) { bin<3>(a, a, o); }
ALWI void sinit() { MATH((SFPU_BINARY_INIT(unused))); }
ALWI void split(uint32_t x, uint32_t hi, uint32_t lo) {
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_pre_split, (0xFFFFE000u, RIT), x, hi, lo,
                           VectorMode::R)));
}
ALWI void rbf16(uint32_t x) {
    MATH((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_pre_rbf16, (RIT), x, VectorMode::R)));
}
ALWI void drain() {
#ifdef TRISC_MATH
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
#endif
}
// The copy / pack configuration is redone only when the source / output kind changes (every copy source is an fp32
// unpack-to-dest CB but the bf16 input, every output an fp32 CB but x and the bf16 outputs; all of one kind share the
// format and tile shape), and a copy after a matmul re-inits.
constexpr uint32_t CB_IN = get_compile_time_arg_val(0), CB_XO = get_compile_time_arg_val(11);
constexpr uint32_t CB_OUT = get_compile_time_arg_val(26), OUT_BF = get_compile_time_arg_val(31);
uint32_t cp_kind = 2, pk_kind = 2;
ALWI void cp_prep(uint32_t cb) {
    drain();
    const uint32_t k = cb == CB_IN;
    if (k != cp_kind) {
        reconfig_full_operand_srca(cb);  // (token-row CBs: TH x 32 tiles; matmul in1: 32 x 32)
        copy_init(cb);
        cp_kind = k;
    }
}
ALWI void cp(uint32_t cb, uint32_t i, uint32_t d) {
    drain();
    const uint32_t k = cb == CB_IN;
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
    const uint32_t k = cb == CB_XO || (OUT_BF && cb == CB_OUT);
    if (k != pk_kind) {
        pack_reconfig_data_format(cb);
        pk_kind = k;
    }
}
ALWI void pack_to(uint32_t d, uint32_t cb) {
    pack_cfg(cb);
    pack_tile(d, cb);
}
// (SFPU_ROWS: the scalar ops and the rsqrt on the live rows only, VectorMode::R x RIT iterations: the same per-element
// code as the whole-tile calls; rows past ROWS are never read)
#ifndef SFPU_ROWS
#define SFPU_ROWS 0
#endif
ALWI void rmul_s(uint32_t d, uint32_t s) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_binop_with_scalar,
                         (APPROX, ckernel::MUL_UNARY, RIT, DST_ACCUM_MODE), d, VectorMode::R, s));
}
ALWI void radd_s(uint32_t d, uint32_t s) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_binop_with_scalar,
                         (APPROX, ckernel::ADD_UNARY, RIT, DST_ACCUM_MODE), d, VectorMode::R, s));
}
ALWI void rrsqrt(uint32_t d) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_rsqrt, (APPROX, RIT, DST_ACCUM_MODE, false, false), d,
                         VectorMode::R));
}
ALWI void rmax(uint32_t a, uint32_t b, uint32_t o) {
    if constexpr (SFPU_ROWS) {
        MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_binary_max_min, (true, RIT), a, b, o,
                               VectorMode::R)));
    } else {
        binary_max_tile(a, b, o, VectorMode::R);
    }
}
ALWI void rrecip(uint32_t d) {
    if constexpr (SFPU_ROWS) {
        MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_reciprocal, (APPROX, DST_ACCUM_MODE, RIT, true), d,
                             VectorMode::R));
    } else {
        recip_tile(d, VectorMode::R);
    }
}
// rsqrt(total / H + eps) in DST d
ALWI void rnorm(uint32_t d, uint32_t inv_h, uint32_t eps) {
    drain();
    binop_with_scalar_tile_init();
    if constexpr (SFPU_ROWS) {
        rmul_s(d, inv_h);
        radd_s(d, eps);
        rsqrt_tile_init<false>();
        rrsqrt(d);
    } else {
        mul_unary_tile(d, inv_h);
        add_unary_tile(d, eps);
        rsqrt_tile_init<false>();
        rsqrt_tile<false>(d);
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rp = get_compile_time_arg_val(1);
    constexpr uint32_t cb_s = get_compile_time_arg_val(2);
    constexpr uint32_t cb_w = get_compile_time_arg_val(3);
    constexpr uint32_t cb_q = get_compile_time_arg_val(4);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(5);
    constexpr uint32_t cb_d = get_compile_time_arg_val(6);
    constexpr uint32_t cb_blk = get_compile_time_arg_val(7);
    constexpr uint32_t cb_nw = get_compile_time_arg_val(8);
    constexpr uint32_t cb_sel = get_compile_time_arg_val(9);
    constexpr uint32_t cb_so = get_compile_time_arg_val(10);   // conv states -> writer
    constexpr uint32_t cb_xo = get_compile_time_arg_val(11);   // x (bf16) -> writer
    constexpr uint32_t cb_x = get_compile_time_arg_val(12);    // x (fp32, kept)
    constexpr uint32_t cb_sq = get_compile_time_arg_val(13);   // per-tile hi / lo products (matmul operands)
    constexpr uint32_t cb_sum = get_compile_time_arg_val(14);  // partial sums (fp32)
    constexpr uint32_t cb_part = get_compile_time_arg_val(15);
    constexpr uint32_t cb_gath = get_compile_time_arg_val(16);
    constexpr uint32_t cb_tout = get_compile_time_arg_val(17);
    constexpr uint32_t cb_tot = get_compile_time_arg_val(18);
    constexpr uint32_t cb_red = get_compile_time_arg_val(19);  // r, lg_j
    constexpr uint32_t cb_wt = get_compile_time_arg_val(20);   // [q][source] weights
    constexpr uint32_t cb_h = get_compile_time_arg_val(21);    // [q][tile] mixes (fp32)
    constexpr uint32_t cb_xn = get_compile_time_arg_val(22);   // xn -> writer
    constexpr uint32_t cb_part2 = get_compile_time_arg_val(23);
    constexpr uint32_t cb_gath2 = get_compile_time_arg_val(24);
    constexpr uint32_t cb_tot2 = get_compile_time_arg_val(25);
    constexpr uint32_t cb_out = get_compile_time_arg_val(26);
    constexpr uint32_t PER = get_compile_time_arg_val(27);
    constexpr uint32_t NB = get_compile_time_arg_val(28);
    constexpr uint32_t NQ = get_compile_time_arg_val(29);
    constexpr uint32_t TAIL = get_compile_time_arg_val(30);
    constexpr uint32_t NORM = get_compile_time_arg_val(31);
    constexpr uint32_t BOUND = get_compile_time_arg_val(32);
    constexpr uint32_t NSEL = get_compile_time_arg_val(33);
    constexpr uint32_t inv_h = get_compile_time_arg_val(34);
    constexpr uint32_t eps = get_compile_time_arg_val(35);
    constexpr uint32_t n = NB + 1;
    constexpr uint32_t K1 = 1 + NQ;  // round 1 quantities: sum x^2, x . q_j
    static_assert(n <= 6 && NQ <= 3, "DST budget");
#ifndef PRE_FUSE
#define PRE_FUSE 0
#endif
    // PRE_FUSE (NQ = 1): round 1's tail with the softmax (n <= 5 leaves DST 6 for r), round 2's tail with the normed
    // output (PER <= 2) in one DST block each
    constexpr bool PFUSE = PRE_FUSE && SOFT_BATCH && NQ == 1 && NB <= 4 && PRE_NP1 == 1;
    constexpr bool PFUSE2 = PRE_FUSE && NQ == 1 && PER <= 2 && PRE_RN1;
#ifndef PRE_CSQ
#define PRE_CSQ 0
#endif
    constexpr bool MSQ = PRE_FUSE && PRE_P1B && NORM;  // (the mix block also squares h: P1B's round-2 layout)
    constexpr bool CSQ = PRE_CSQ && NQ == 1;  // (round 1's products in the x block; the q rows read with the x inputs)
    // core t: the root of row t (PRE_PACK: core 0 of every row, the selection tile's row t picking row t's partials)
    const bool root = PRE_PACK ? get_arg_val<uint32_t>(0) == 0 : get_arg_val<uint32_t>(0) < ROWS;
    const uint32_t ci = get_arg_val<uint32_t>(0);
    bool sw = false;  // (waited for the selection tiles: pops them at the end)
    // a round's totals of npair hi / lo pairs: all on root t (NP = 1) or pair j on core t NP + j (the same matmuls)
    auto pairs = [&](uint32_t npair, uint32_t NP) {
        if (!(NP > 1 ? ci < ROWS * NP : root)) {
            return;
        }
        cb_wait_front(cb_sel, NSEL);
        sw = true;
        tile_regs_acquire();
        mm_begin(cb_sel, cb_gath);  // (configured before the gather arrives)
        cb_wait_front(cb_gath, 2 * K1 * NSEL);  // (the shared gather buffer, pushed whole)
        const uint32_t k0 = NP > 1 ? ci % NP : 0, nk = NP > 1 ? 1 : npair;
        for (uint32_t k = k0; k < k0 + nk; ++k) {
            for (uint32_t s = 0; s < NSEL; ++s) {
                matmul_tiles(cb_sel, cb_gath, s, (2 * k) * NSEL + s, k - k0);
                matmul_tiles(cb_sel, cb_gath, s, (2 * k + 1) * NSEL + s, k - k0);
            }
        }
        tile_regs_commit();
        cb_reserve_back(cb_tout, nk);
        tile_regs_wait();
        pack_cfg(cb_tout);
        for (uint32_t k = 0; k < nk; ++k) {
            pack_tile(k, cb_tout);
        }
        tile_regs_release();
        cb_push_back(cb_tout, nk);
        GLZ("C_TOUT");
        cb_pop_front(cb_gath, 2 * K1 * NSEL);
    };

    // matmuls: in0 -> srcB, in1 -> srcA (each CB's tile shape: the token-row CBs hold TH x 32 tiles, PRE_TH)
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_sq, cb_ones, cb_so);

    // ---- x (TAIL: the previous application's conv + residual); CSQ: with round 1's products in the same DST block
    cb_reserve_back(cb_x, PER);
    if constexpr (CSQ) {
        cb_wait_front(cb_q, PER);
        cb_reserve_back(cb_sq, 2 * K1 * PER);
    }
    for (uint32_t i = 0; i < PER; ++i) {
        cb_wait_front(cb_in, 1);
        GLZ("C_IN");
        tile_regs_acquire();
        if constexpr (TAIL) {
            cb_wait_front(cb_s, 3);
            cb_wait_front(cb_w, 4);
            GLZ("C_IN");
            cb_wait_front(cb_rp, 1);
            cp(cb_in, 0, 0);
            cp(cb_w, 3, 1);
            copy_tile(cb_s, 0, 2);
            copy_tile(cb_w, 0, 3);
            copy_tile(cb_s, 1, 4);
            copy_tile(cb_w, 1, 5);
            copy_tile(cb_s, 2, 6);
            copy_tile(cb_w, 2, 7);
            sinit();
            mul(0, 1, 1);
            mul(2, 3, 3);
            add(1, 3, 1);
            mul(4, 5, 5);
            add(1, 5, 1);
            mul(6, 7, 7);
            add(1, 7, 1);
            rbf16(1);
            cp(cb_rp, 0, 2);
            sinit();
            add(1, 2, 1);
            rbf16(1);  // x = bf16(resid + y)
        } else {
            cp(cb_in, 0, 1);
        }
        if constexpr (CSQ) {  // (DST 2, 3, 5, 7 free: q at 7; [x^2 hi, lo, x q hi, lo] -> 2, 3, 5, 7)
            cp(cb_q, i, 7);
            sinit();
            mul(1, 7, 7);
            mul(1, 1, 5);
            split(5, 2, 3);
            split(7, 5, 7);
        }
        tile_regs_commit();
        tile_regs_wait();
        if constexpr (TAIL) {
            cb_reserve_back(cb_so, 3);
            cb_reserve_back(cb_xo, 1);
            pack_to(4, cb_so);
            pack_tile(6, cb_so);
            pack_tile(0, cb_so);
            pack_to(1, cb_xo);
        }
        pack_to(1, cb_x);
        if constexpr (CSQ) {
            pack_tile(2, cb_sq);
            pack_tile(3, cb_sq);
            pack_tile(5, cb_sq);
            pack_tile(7, cb_sq);
        }
        tile_regs_release();
        if constexpr (TAIL) {
            cb_push_back(cb_so, 3);
            cb_push_back(cb_xo, 1);
            cb_pop_front(cb_s, 3);
            cb_pop_front(cb_w, 4);
            cb_pop_front(cb_rp, 1);
        }
        cb_pop_front(cb_in, 1);
    }
    cb_push_back(cb_x, PER);
    cb_wait_front(cb_x, PER);

    // ---- round 1 partials: [x^2 | x q_0 | ...] hi / lo products per tile -> sums -> hi / lo parts
    cb_wait_front(cb_ones, 1);
    cb_wait_front(cb_q, NQ * PER);
    GLZ("C_IN");
    if constexpr (CSQ) {  // (the products per tile [k hi, k lo] for k < K1: the same sums as the per-quantity passes)
        cb_push_back(cb_sq, 2 * K1 * PER);
        cb_wait_front(cb_sq, 2 * K1 * PER);
        tile_regs_acquire();
        mm_begin(cb_sq, cb_ones);
        for (uint32_t k = 0; k < K1; ++k) {
            for (uint32_t i = 0; i < PER; ++i) {
                matmul_tiles(cb_sq, cb_ones, i * 2 * K1 + 2 * k, 0, k);
                matmul_tiles(cb_sq, cb_ones, i * 2 * K1 + 2 * k + 1, 0, k);
            }
        }
        sinit();
        for (uint32_t k = 0; k < K1; ++k) {
            split(k, k, 4 + k);
        }
        tile_regs_commit();
        cb_reserve_back(cb_part, 2 * K1);
        tile_regs_wait();
        pack_cfg(cb_part);
        for (uint32_t k = 0; k < K1; ++k) {
            pack_tile(k, cb_part);
            pack_tile(4 + k, cb_part);
        }
        tile_regs_release();
        cb_push_back(cb_part, 2 * K1);
        GLZ("C_PP");
        cb_pop_front(cb_sq, 2 * K1 * PER);
    } else {
#if PRE_P1B  // (every quantity's products, then one matmul pass: the same ops, one hand-off instead of K1)
    cb_reserve_back(cb_sq, 2 * K1 * PER);
    for (uint32_t k = 0; k < K1; ++k) {
        for (uint32_t i = 0; i < PER; ++i) {
            tile_regs_acquire();
            cp(cb_x, i, 0);
            if (k == 0) {
                sinit();
                mul(0, 0, 0);
            } else {
                copy_tile(cb_q, (k - 1) * PER + i, 1);
                sinit();
                mul(0, 1, 0);
            }
            split(0, 2, 3);
            tile_regs_commit();
            tile_regs_wait();
            pack_to(2, cb_sq);
            pack_tile(3, cb_sq);
            tile_regs_release();
        }
    }
    cb_push_back(cb_sq, 2 * K1 * PER);
    cb_wait_front(cb_sq, 2 * K1 * PER);
    tile_regs_acquire();
    mm_begin(cb_sq, cb_ones);
    for (uint32_t k = 0; k < K1; ++k) {
        for (uint32_t i = 0; i < 2 * PER; ++i) {
            matmul_tiles(cb_sq, cb_ones, k * 2 * PER + i, 0, k);
        }
    }
    sinit();
    for (uint32_t k = 0; k < K1; ++k) {
        split(k, k, 4 + k);
    }
    tile_regs_commit();
    cb_reserve_back(cb_part, 2 * K1);
    tile_regs_wait();
    pack_cfg(cb_part);
    for (uint32_t k = 0; k < K1; ++k) {
        pack_tile(k, cb_part);
        pack_tile(4 + k, cb_part);
    }
    tile_regs_release();
    cb_push_back(cb_part, 2 * K1);
    GLZ("C_PP");
    cb_pop_front(cb_sq, 2 * K1 * PER);
#else
    for (uint32_t k = 0; k < K1; ++k) {
        cb_reserve_back(cb_sq, 2 * PER);
        for (uint32_t i = 0; i < PER; ++i) {
            tile_regs_acquire();
            cp(cb_x, i, 0);
            if (k == 0) {
                sinit();
                mul(0, 0, 0);
            } else {
                copy_tile(cb_q, (k - 1) * PER + i, 1);
                sinit();
                mul(0, 1, 0);
            }
            split(0, 2, 3);
            tile_regs_commit();
            tile_regs_wait();
            pack_to(2, cb_sq);
            pack_tile(3, cb_sq);
            tile_regs_release();
        }
        cb_push_back(cb_sq, 2 * PER);
        cb_wait_front(cb_sq, 2 * PER);
        tile_regs_acquire();
        mm_begin(cb_sq, cb_ones);
        for (uint32_t i = 0; i < 2 * PER; ++i) {
            matmul_tiles(cb_sq, cb_ones, i, 0, 0);
        }
        sinit();
        split(0, 2, 3);
        tile_regs_commit();
        cb_reserve_back(cb_part, 2);
        tile_regs_wait();
        pack_to(2, cb_part);
        pack_tile(3, cb_part);
        tile_regs_release();
        cb_push_back(cb_part, 2);
        GLZ("C_PP");
        cb_pop_front(cb_sq, 2 * PER);
    }
#endif
    }
    cb_pop_front(cb_q, NQ * PER);

    // ---- root: totals of round 1 (gather tiles [quantity][hi, lo][NSEL])
    pairs(K1, PRE_NP1);
    // ---- r, lg_j
    if constexpr (PFUSE) {  // (NQ = 1: r in DST 6, lg at n and copied to 0; the round-1 tail and the softmax in one block)
        cb_wait_front(cb_d, NB);
        tile_regs_acquire();
        cp_prep(cb_tot);  // (configured before the totals arrive)
        cb_wait_front(cb_tot, K1);
        GLZ("C_TOT");
        cp(cb_tot, 0, 6);
        cp(cb_tot, 1, n);
        rnorm(6, inv_h, eps);
        sinit();
        mul(n, 6, n);  // lg = r dot
        mov(n, 0);
        for (uint32_t i = 0; i < NB; ++i) {
            cp(cb_d, i, 1 + i);
        }
        if constexpr (NB > 0) {
            drain();
            binary_max_tile_init();
            for (uint32_t b = 0; b < NB; ++b) {
                rmax(0, 1 + b, 0);
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
        if constexpr (n > 1) {
            drain();
            sinit();
            add(1, 2, 7);
            for (uint32_t i = 2; i < n; ++i) {
                add(7, 1 + i, 7);
            }
            drain();
            recip_tile_init();
            rrecip(7);
            sinit();
            for (uint32_t i = 0; i < n; ++i) {
                mul(1 + i, 7, 1 + i);
            }
        }
        sinit();
        mul(n, 6, n);  // (the current source's weight times r)
        tile_regs_commit();
        cb_reserve_back(cb_wt, n);
        cb_reserve_back(cb_red, 1);
        tile_regs_wait();
        pack_cfg(cb_wt);
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(1 + i, cb_wt);
        }
        pack_tile(6, cb_red);  // (r: the boundary's xn)
        tile_regs_release();
        cb_push_back(cb_wt, n);
        cb_push_back(cb_red, 1);
        cb_pop_front(cb_tot, K1);
        cb_wait_front(cb_wt, n);
        cb_wait_front(cb_red, 1);
        cb_pop_front(cb_d, NB);
    } else {
        tile_regs_acquire();
        cp_prep(cb_tot);  // (configured before the totals arrive)
        cb_wait_front(cb_tot, K1);
        GLZ("C_TOT");
        cp(cb_tot, 0, 0);
        for (uint32_t q = 0; q < NQ; ++q) {
            copy_tile(cb_tot, 1 + q, 1 + q);
        }
        rnorm(0, inv_h, eps);
        sinit();
        for (uint32_t q = 0; q < NQ; ++q) {
            mul(1 + q, 0, 1 + q);
        }
        tile_regs_commit();
        cb_reserve_back(cb_red, K1);
        tile_regs_wait();
        pack_cfg(cb_red);
        for (uint32_t k = 0; k < K1; ++k) {
            pack_tile(k, cb_red);
        }
        tile_regs_release();
        cb_push_back(cb_red, K1);
        cb_pop_front(cb_tot, K1);
        cb_wait_front(cb_red, K1);

        // ---- softmax per query over [block logits (cb_d [b][q])..., lg_q]; the current weight times r
        cb_wait_front(cb_d, NB * NQ);
        GLZ("C_IN");
        cb_reserve_back(cb_wt, NQ * n);
        for (uint32_t q = 0; q < NQ; ++q) {
            tile_regs_acquire();
    #if SOFT_BATCH  // (each SFPU op over all sources: one init per op kind)
            cp(cb_red, 1 + q, 0);
            for (uint32_t i = 0; i < n; ++i) {
                if (i < NB) {
                    cp(cb_d, i * NQ + q, 1 + i);
                } else {
                    cp(cb_red, 1 + q, 1 + i);
                }
            }
            if constexpr (NB > 0) {
                drain();
                binary_max_tile_init();
                for (uint32_t b = 0; b < NB; ++b) {
                    rmax(0, 1 + b, 0);
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
            cp(cb_red, 1 + q, 0);
            for (uint32_t b = 0; b < NB; ++b) {
                cp(cb_d, b * NQ + q, 1);
                binary_max_tile_init();
                rmax(0, 1, 0);
            }
            for (uint32_t i = 0; i < n; ++i) {
                if (i < NB) {
                    cp(cb_d, i * NQ + q, 1 + i);
                } else {
                    cp(cb_red, 1 + q, 1 + i);
                }
                sinit();
                sub(1 + i, 0, 1 + i);
                exp_tile_init();
                exp_tile<false, false, InputClamping::ClampToNegative, RIT>(1 + i, VectorMode::R);
            }
    #endif
            if constexpr (n > 1) {
                drain();
                sinit();
                add(1, 2, 7);
                for (uint32_t i = 2; i < n; ++i) {
                    add(7, 1 + i, 7);
                }
                drain();
                recip_tile_init();
                rrecip(7);
                sinit();
                for (uint32_t i = 0; i < n; ++i) {
                    mul(1 + i, 7, 1 + i);
                }
            }
            cp(cb_red, 0, 7);
            sinit();
            mul(n, 7, n);
            tile_regs_commit();
            tile_regs_wait();
            pack_cfg(cb_wt);
            for (uint32_t i = 0; i < n; ++i) {
                pack_tile(1 + i, cb_wt);
            }
            tile_regs_release();
        }
        cb_push_back(cb_wt, NQ * n);
        cb_wait_front(cb_wt, NQ * n);
        cb_pop_front(cb_d, NB * NQ);
    }

    // ---- mixes h_q (and xn = r x at a block boundary)
    cb_wait_front(cb_blk, NB * PER);
    GLZ("C_IN");
    cb_reserve_back(cb_h, NQ * PER);
    if constexpr (MSQ) {
        cb_reserve_back(cb_sq, 2 * NQ * PER);
    }
    for (uint32_t q = 0; q < NQ; ++q) {
        for (uint32_t i = 0; i < PER; ++i) {
            tile_regs_acquire();
            cp(cb_x, i, 1);
            copy_tile(cb_wt, q * n + NB, 2);
            sinit();
            mul(1, 2, 0);
            for (uint32_t b = 0; b < NB; ++b) {
                const uint32_t d = 3 + 2 * (b % 2);
                cp(cb_blk, b * PER + i, d);
                copy_tile(cb_wt, q * n + b, d + 1);
                sinit();
                mul(d, d + 1, d);
                add(0, d, 0);
            }
            if constexpr (MSQ) {  // (round 2's product h^2 hi / lo here: no h round trip)
                sinit();
                mul(0, 0, 1);
                split(1, 1, 2);
            }
            tile_regs_commit();
            tile_regs_wait();
            if constexpr (NORM) {
                pack_to(0, cb_h);
                if constexpr (MSQ) {
                    pack_tile(1, cb_sq);
                    pack_tile(2, cb_sq);
                }
            } else {
                cb_reserve_back(cb_out, 1);
                pack_to(0, cb_out);
                cb_push_back(cb_out, 1);
            }
            tile_regs_release();
        }
    }
    if constexpr (BOUND) {
        for (uint32_t i = 0; i < PER; ++i) {
            tile_regs_acquire();
            cp(cb_x, i, 0);
            cp(cb_red, 0, 1);
            sinit();
            mul(0, 1, 0);
            tile_regs_commit();
            cb_reserve_back(cb_xn, 1);
            tile_regs_wait();
            pack_to(0, cb_xn);
            tile_regs_release();
            cb_push_back(cb_xn, 1);
        }
    }
    cb_pop_front(cb_blk, NB * PER);
    cb_pop_front(cb_x, PER);
    cb_pop_front(cb_wt, NQ * n);
    cb_pop_front(cb_red, PFUSE ? 1 : K1);

    if constexpr (NORM) {
        cb_push_back(cb_h, NQ * PER);
        cb_wait_front(cb_h, NQ * PER);
        // round 2 partials: sum h_q^2
#if PRE_P1B  // (as round 1: every query's products, then one matmul pass)
        if constexpr (!MSQ) {
        cb_reserve_back(cb_sq, 2 * NQ * PER);
        for (uint32_t q = 0; q < NQ; ++q) {
            for (uint32_t i = 0; i < PER; ++i) {
                tile_regs_acquire();
                cp(cb_h, q * PER + i, 0);
                sinit();
                mul(0, 0, 0);
                split(0, 2, 3);
                tile_regs_commit();
                tile_regs_wait();
                pack_to(2, cb_sq);
                pack_tile(3, cb_sq);
                tile_regs_release();
            }
        }
        }
        cb_push_back(cb_sq, 2 * NQ * PER);
        cb_wait_front(cb_sq, 2 * NQ * PER);
        tile_regs_acquire();
        mm_begin(cb_sq, cb_ones);
        for (uint32_t q = 0; q < NQ; ++q) {
            for (uint32_t i = 0; i < 2 * PER; ++i) {
                matmul_tiles(cb_sq, cb_ones, q * 2 * PER + i, 0, q);
            }
        }
        sinit();
        for (uint32_t q = 0; q < NQ; ++q) {
            split(q, q, 4 + q);
        }
        tile_regs_commit();
        cb_reserve_back(cb_part2, 2 * NQ);
        tile_regs_wait();
        pack_cfg(cb_part2);
        for (uint32_t q = 0; q < NQ; ++q) {
            pack_tile(q, cb_part2);
            pack_tile(4 + q, cb_part2);
        }
        tile_regs_release();
        cb_push_back(cb_part2, 2 * NQ);
        GLZ("C_PP");
        cb_pop_front(cb_sq, 2 * NQ * PER);
#else
        for (uint32_t q = 0; q < NQ; ++q) {
            cb_reserve_back(cb_sq, 2 * PER);
            for (uint32_t i = 0; i < PER; ++i) {
                tile_regs_acquire();
                cp(cb_h, q * PER + i, 0);
                sinit();
                mul(0, 0, 0);
                split(0, 2, 3);
                tile_regs_commit();
                tile_regs_wait();
                pack_to(2, cb_sq);
                pack_tile(3, cb_sq);
                tile_regs_release();
            }
            cb_push_back(cb_sq, 2 * PER);
            cb_wait_front(cb_sq, 2 * PER);
            tile_regs_acquire();
            mm_begin(cb_sq, cb_ones);
            for (uint32_t i = 0; i < 2 * PER; ++i) {
                matmul_tiles(cb_sq, cb_ones, i, 0, 0);
            }
            sinit();
            split(0, 2, 3);
            tile_regs_commit();
            cb_reserve_back(cb_part2, 2);
            tile_regs_wait();
            pack_to(2, cb_part2);
            pack_tile(3, cb_part2);
            tile_regs_release();
            cb_push_back(cb_part2, 2);
            GLZ("C_PP");
            cb_pop_front(cb_sq, 2 * PER);
        }
#endif
        pairs(NQ, PRE_NP2);
        if constexpr (PFUSE2) {  // (NQ = 1: rn in DST 7, h at 0.., nw at PER..: one block, no rn round trip)
            cb_wait_front(cb_nw, PER);
            tile_regs_acquire();
            cp_prep(cb_tot2);
            cb_wait_front(cb_tot2, 1);
            GLZ("C_TOT");
            cp(cb_tot2, 0, 7);
            rnorm(7, inv_h, eps);
            for (uint32_t i = 0; i < PER; ++i) {
                cp(cb_h, i, i);
                cp(cb_nw, i, PER + i);
            }
            sinit();
            for (uint32_t i = 0; i < PER; ++i) {
                mul(i, 7, i);
            }
            for (uint32_t i = 0; i < PER; ++i) {
                mul(i, PER + i, i);
            }
            tile_regs_commit();
            cb_reserve_back(cb_out, PER);
            tile_regs_wait();
            pack_cfg(cb_out);
            for (uint32_t i = 0; i < PER; ++i) {
                pack_tile(i, cb_out);
            }
            tile_regs_release();
            cb_push_back(cb_out, PER);
            cb_pop_front(cb_tot2, 1);
            cb_pop_front(cb_nw, PER);
            cb_pop_front(cb_h, PER);
        } else {
    #if PRE_RN1  // (each query's rsqrt once, through cb_red: free again after the mixes)
            tile_regs_acquire();
            cp_prep(cb_tot2);
            cb_wait_front(cb_tot2, NQ);
            GLZ("C_TOT");
            for (uint32_t q = 0; q < NQ; ++q) {
                cp(cb_tot2, q, q);
                rnorm(q, inv_h, eps);
            }
            tile_regs_commit();
            cb_reserve_back(cb_red, NQ);
            tile_regs_wait();
            pack_cfg(cb_red);
            for (uint32_t q = 0; q < NQ; ++q) {
                pack_tile(q, cb_red);
            }
            tile_regs_release();
            cb_push_back(cb_red, NQ);
            cb_wait_front(cb_red, NQ);
    #endif
            cb_wait_front(cb_tot2, NQ);
            GLZ("C_TOT");
            cb_wait_front(cb_nw, NQ * PER);
            GLZ("C_IN");
            for (uint32_t q = 0; q < NQ; ++q) {
                for (uint32_t i = 0; i < PER; ++i) {
                    tile_regs_acquire();
    #if PRE_RN1
                    cp(cb_red, q, 1);
    #else
                    cp(cb_tot2, q, 1);
                    rnorm(1, inv_h, eps);
    #endif
                    cp(cb_h, q * PER + i, 0);
                    sinit();
                    mul(0, 1, 0);
                    cp(cb_nw, q * PER + i, 1);
                    sinit();
                    mul(0, 1, 0);
                    tile_regs_commit();
                    cb_reserve_back(cb_out, 1);
                    tile_regs_wait();
                    pack_to(0, cb_out);
                    tile_regs_release();
                    cb_push_back(cb_out, 1);
                }
            }
            cb_pop_front(cb_tot2, NQ);
    #if PRE_RN1
            cb_pop_front(cb_red, NQ);
    #endif
            cb_pop_front(cb_nw, NQ * PER);
            cb_pop_front(cb_h, NQ * PER);
        }
    }
    cb_pop_front(cb_ones, 1);
    if (sw) {
        cb_pop_front(cb_sel, NSEL);
    }
}
