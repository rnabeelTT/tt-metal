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
        } else {
            sfpi::dst_reg[o * tile] = x * y;
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
// rsqrt(total / H + eps) in DST d
ALWI void rnorm(uint32_t d, uint32_t inv_h, uint32_t eps) {
    drain();
    binop_with_scalar_tile_init();
    mul_unary_tile(d, inv_h);
    add_unary_tile(d, eps);
    rsqrt_tile_init<false>();
    rsqrt_tile<false>(d);
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
    const bool root = get_arg_val<uint32_t>(0) < ROWS;  // core t: the root of row t

    // matmuls: in0 -> srcB, in1 -> srcA (each CB's tile shape: the token-row CBs hold TH x 32 tiles, PRE_TH)
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_sq, cb_ones, cb_so);

    // ---- x (TAIL: the previous application's conv + residual)
    cb_reserve_back(cb_x, PER);
    for (uint32_t i = 0; i < PER; ++i) {
        cb_wait_front(cb_in, 1);
        tile_regs_acquire();
        if constexpr (TAIL) {
            cb_wait_front(cb_s, 3);
            cb_wait_front(cb_w, 4);
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
        cb_pop_front(cb_sq, 2 * PER);
    }
    cb_pop_front(cb_q, NQ * PER);

    // ---- root: totals of round 1 (gather tiles [quantity][hi, lo][NSEL])
    if (root) {
        cb_wait_front(cb_sel, NSEL);
        tile_regs_acquire();
        mm_begin(cb_sel, cb_gath);  // (configured before the gather arrives)
        cb_wait_front(cb_gath, 2 * K1 * NSEL);
        for (uint32_t k = 0; k < K1; ++k) {
            for (uint32_t s = 0; s < NSEL; ++s) {
                matmul_tiles(cb_sel, cb_gath, s, (2 * k) * NSEL + s, k);
                matmul_tiles(cb_sel, cb_gath, s, (2 * k + 1) * NSEL + s, k);
            }
        }
        tile_regs_commit();
        cb_reserve_back(cb_tout, K1);
        tile_regs_wait();
        pack_cfg(cb_tout);
        for (uint32_t k = 0; k < K1; ++k) {
            pack_tile(k, cb_tout);
        }
        tile_regs_release();
        cb_push_back(cb_tout, K1);
        cb_pop_front(cb_gath, 2 * K1 * NSEL);
    }
    // ---- r, lg_j
    tile_regs_acquire();
    cp_prep(cb_tot);  // (configured before the totals arrive)
    cb_wait_front(cb_tot, K1);
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
        cp(cb_red, 1 + q, 0);
        for (uint32_t b = 0; b < NB; ++b) {
            cp(cb_d, b * NQ + q, 1);
            binary_max_tile_init();
            binary_max_tile(0, 1, 0, VectorMode::R);
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
            recip_tile(7, VectorMode::R);
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

    // ---- mixes h_q (and xn = r x at a block boundary)
    cb_wait_front(cb_blk, NB * PER);
    cb_reserve_back(cb_h, NQ * PER);
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
            tile_regs_commit();
            tile_regs_wait();
            if constexpr (NORM) {
                pack_to(0, cb_h);
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
    cb_pop_front(cb_red, K1);

    if constexpr (NORM) {
        cb_push_back(cb_h, NQ * PER);
        cb_wait_front(cb_h, NQ * PER);
        // round 2 partials: sum h_q^2
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
            cb_pop_front(cb_sq, 2 * PER);
        }
        if (root) {
            tile_regs_acquire();
            mm_begin(cb_sel, cb_gath);  // (configured before the gather arrives)
            cb_wait_front(cb_gath, 2 * K1 * NSEL);  // (the shared gather buffer, pushed whole)
            for (uint32_t q = 0; q < NQ; ++q) {
                for (uint32_t s = 0; s < NSEL; ++s) {
                    matmul_tiles(cb_sel, cb_gath, s, (2 * q) * NSEL + s, q);
                    matmul_tiles(cb_sel, cb_gath, s, (2 * q + 1) * NSEL + s, q);
                }
            }
            tile_regs_commit();
            cb_reserve_back(cb_tout, NQ);
            tile_regs_wait();
            pack_cfg(cb_tout);
            for (uint32_t q = 0; q < NQ; ++q) {
                pack_tile(q, cb_tout);
            }
            tile_regs_release();
            cb_push_back(cb_tout, NQ);
            cb_pop_front(cb_gath, 2 * K1 * NSEL);
        }
#if PRE_RN1  // (each query's rsqrt once, through cb_red: free again after the mixes)
        tile_regs_acquire();
        cp_prep(cb_tot2);
        cb_wait_front(cb_tot2, NQ);
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
        cb_wait_front(cb_nw, NQ * PER);
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
    cb_pop_front(cb_ones, 1);
    if (root) {
        cb_pop_front(cb_sel, NSEL);
    }
}
