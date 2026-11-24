// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Message gate, round-reduced (tt/gate.py GateKernel2), compute (live rows, VectorMode::R):
//   r_h, r_m = rsqrt(mean(h^2) + eps), rsqrt(mean(m^2) + eps)           (round 1: exact hi / lo sums)
//   g = bf16(Wd [bf16(r_h h) | bf16(r_m m)])                           (round 2: partial projections, exact sums)
//   out = h - tanh(Wf g) h + tanh(Wg g) m                              (local, fp32)
//   NORM: out = bf16(out rsqrt(mean(out^2) + eps) nw)                  (round 3)
//   ROUTER: logits = out @ (W hi + W lo) (bf16 halves of the fp32 router weight; round 4, the root keeps the totals)
// Every round's gather is the same buffer (GP parts of NSEL tiles, pushed / popped whole).

#include <cstdint>

#include "api/compute/binary_max_min.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/compute_kernel_api.h"
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
// The copy / pack configuration is redone only when needed: every copy source is an fp32 unpack-to-dest CB of token
// rows (one format and tile shape), so a copy re-inits only after a matmul; a pack reconfigures when the output kind
// (fp32 / bf16 token-row tiles) changes.
constexpr uint32_t CB_HN = get_compile_time_arg_val(13), CB_G = get_compile_time_arg_val(16);
constexpr uint32_t CB_OUT = get_compile_time_arg_val(18), OUT_BF = get_compile_time_arg_val(25);
constexpr uint32_t CB_XB = get_compile_time_arg_val(31);
uint32_t cp_kind = 2, pk_kind = 2;
ALWI void cp_prep(uint32_t cb) {
    drain();
    if (cp_kind != 0) {
        reconfig_full_operand_srca(cb);  // (token-row CBs: TH x 32 tiles; matmul in1: 32 x 32)
        copy_init(cb);
        cp_kind = 0;
    }
}
ALWI void cp(uint32_t cb, uint32_t i, uint32_t d) {
    drain();
    if (cp_kind != 0) {
        reconfig_full_operand_srca(cb);  // (token-row CBs: TH x 32 tiles; matmul in1: 32 x 32)
        copy_init(cb);
        cp_kind = 0;
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
    const uint32_t k = cb == CB_HN || cb == CB_G || cb == CB_XB || (OUT_BF && cb == CB_OUT);
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

ALWI void rtanh(uint32_t i) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_tanh, (false, DST_ACCUM_MODE, RIT), i, VectorMode::R));
}

void kernel_main() {
    constexpr uint32_t cb_h = get_compile_time_arg_val(0);
    constexpr uint32_t cb_m = get_compile_time_arg_val(1);
    constexpr uint32_t cb_wd = get_compile_time_arg_val(2);
    constexpr uint32_t cb_wf = get_compile_time_arg_val(3);
    constexpr uint32_t cb_wg = get_compile_time_arg_val(4);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(5);
    constexpr uint32_t cb_sel = get_compile_time_arg_val(6);
    constexpr uint32_t cb_sq = get_compile_time_arg_val(7);
    constexpr uint32_t cb_part = get_compile_time_arg_val(8);
    constexpr uint32_t cb_gath = get_compile_time_arg_val(9);
    constexpr uint32_t cb_tout = get_compile_time_arg_val(10);
    constexpr uint32_t cb_tot = get_compile_time_arg_val(11);
    constexpr uint32_t cb_r = get_compile_time_arg_val(12);
    constexpr uint32_t cb_hn = get_compile_time_arg_val(13);
    constexpr uint32_t cb_gath2 = get_compile_time_arg_val(14);
    constexpr uint32_t cb_tot2 = get_compile_time_arg_val(15);
    constexpr uint32_t cb_g = get_compile_time_arg_val(16);
    constexpr uint32_t cb_fa = get_compile_time_arg_val(17);
    constexpr uint32_t cb_out = get_compile_time_arg_val(18);
    constexpr uint32_t PER = get_compile_time_arg_val(19);
    constexpr uint32_t R = get_compile_time_arg_val(20);
    constexpr uint32_t NSEL = get_compile_time_arg_val(21);
    constexpr uint32_t inv_h = get_compile_time_arg_val(22);
    constexpr uint32_t eps = get_compile_time_arg_val(23);
    constexpr uint32_t GP = get_compile_time_arg_val(24);
    constexpr uint32_t NORM = get_compile_time_arg_val(25);
    constexpr uint32_t ROUTER = get_compile_time_arg_val(26);
    constexpr uint32_t ET = get_compile_time_arg_val(27);
    constexpr uint32_t cb_o32 = get_compile_time_arg_val(28);
    constexpr uint32_t cb_tot3 = get_compile_time_arg_val(29);
    constexpr uint32_t cb_nw = get_compile_time_arg_val(30);
    constexpr uint32_t cb_xb = get_compile_time_arg_val(31);
    constexpr uint32_t cb_rw = get_compile_time_arg_val(32);
    constexpr uint32_t GT = GP * NSEL;
    static_assert(ET <= 4, "router tiles");
    static_assert(R == 2, "rank tiles");
    const bool root = get_arg_val<uint32_t>(0) < ROWS;  // core t: the root of row t

    // matmuls: in0 -> srcB, in1 -> srcA (each CB's tile shape: the token-row CBs hold TH x 32 tiles, G2_TH)
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_sq, cb_ones, cb_part);
    DeviceZoneScopedN("G2_C_ALL");
    cb_wait_front(cb_h, PER);
    cb_wait_front(cb_m, PER);
    cb_wait_front(cb_ones, 1);

    // ---- round 1 partials: sum h^2, sum m^2 (hi / lo products, exact) -> hi / lo parts [h hi, h lo, m hi, m lo]
    cb_reserve_back(cb_sq, 4 * PER);
    for (uint32_t i = 0; i < PER; ++i) {
        tile_regs_acquire();
        cp(cb_h, i, 0);
        copy_tile(cb_m, i, 1);
        sinit();
        mul(0, 0, 0);
        mul(1, 1, 1);
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
    tile_regs_acquire();
    mm_begin(cb_sq, cb_ones);
    for (uint32_t i = 0; i < PER; ++i) {
        matmul_tiles(cb_sq, cb_ones, 4 * i, 0, 0);
        matmul_tiles(cb_sq, cb_ones, 4 * i + 1, 0, 0);
        matmul_tiles(cb_sq, cb_ones, 4 * i + 2, 0, 1);
        matmul_tiles(cb_sq, cb_ones, 4 * i + 3, 0, 1);
    }
    sinit();
    split(0, 2, 3);
    split(1, 4, 5);
    tile_regs_commit();
    cb_reserve_back(cb_part, 4);
    tile_regs_wait();
    pack_to(2, cb_part);
    pack_tile(3, cb_part);
    pack_tile(4, cb_part);
    pack_tile(5, cb_part);
    tile_regs_release();
    cb_push_back(cb_part, 4);
    cb_pop_front(cb_sq, 4 * PER);

    if (root) {  // totals [sum h^2, sum m^2] from the gathered parts
        cb_wait_front(cb_sel, NSEL);
        tile_regs_acquire();
        mm_begin(cb_sel, cb_gath);  // (configured before the gather arrives)
        cb_wait_front(cb_gath, GT);
        for (uint32_t k = 0; k < 2; ++k) {
            for (uint32_t s = 0; s < NSEL; ++s) {
                matmul_tiles(cb_sel, cb_gath, s, (2 * k) * NSEL + s, k);
                matmul_tiles(cb_sel, cb_gath, s, (2 * k + 1) * NSEL + s, k);
            }
        }
        tile_regs_commit();
        cb_reserve_back(cb_tout, 2);
        tile_regs_wait();
        pack_to(0, cb_tout);
        pack_tile(1, cb_tout);
        tile_regs_release();
        cb_push_back(cb_tout, 2);
        cb_pop_front(cb_gath, GT);
    }
    tile_regs_acquire();
    cp_prep(cb_tot);  // (configured before the totals arrive)
    cb_wait_front(cb_tot, 2);
    cp(cb_tot, 0, 0);
    copy_tile(cb_tot, 1, 1);
    rnorm(0, inv_h, eps);
    rnorm(1, inv_h, eps);
    tile_regs_commit();
    cb_reserve_back(cb_r, 2);
    tile_regs_wait();
    pack_to(0, cb_r);
    pack_tile(1, cb_r);
    tile_regs_release();
    cb_push_back(cb_r, 2);
    cb_pop_front(cb_tot, 2);
    cb_wait_front(cb_r, 2);

    // ---- normed bf16 operands [h tiles | m tiles]
    cb_reserve_back(cb_hn, 2 * PER);
    for (uint32_t which = 0; which < 2; ++which) {
        const uint32_t cbx = which ? cb_m : cb_h;
        for (uint32_t i = 0; i < PER; ++i) {
            tile_regs_acquire();
            cp(cbx, i, 0);
            cp(cb_r, which, 1);
            sinit();
            mul(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_to(0, cb_hn);
            tile_regs_release();
        }
    }
    cb_push_back(cb_hn, 2 * PER);
    cb_pop_front(cb_r, 2);
    cb_wait_front(cb_hn, 2 * PER);
    cb_wait_front(cb_wd, 2 * PER * R);

    // ---- round 2 partials: [hn | mn] @ [Wd h rows | Wd m rows] -> R tiles -> hi / lo parts [g0 hi, lo, g1 hi, lo]
    tile_regs_acquire();
    mm_begin(cb_hn, cb_wd);
    for (uint32_t nn = 0; nn < R; ++nn) {
        for (uint32_t i = 0; i < 2 * PER; ++i) {
            matmul_tiles(cb_hn, cb_wd, i, i * R + nn, nn);
        }
    }
    sinit();
    split(0, 2, 3);
    split(1, 4, 5);
    tile_regs_commit();
    cb_reserve_back(cb_part, 4);
    tile_regs_wait();
    pack_to(2, cb_part);
    pack_tile(3, cb_part);
    pack_tile(4, cb_part);
    pack_tile(5, cb_part);
    tile_regs_release();
    cb_push_back(cb_part, 4);
    cb_pop_front(cb_hn, 2 * PER);
    cb_pop_front(cb_wd, 2 * PER * R);

    if (root) {
        tile_regs_acquire();
        mm_begin(cb_sel, cb_gath);  // (configured before the gather arrives)
        cb_wait_front(cb_gath, GT);
        for (uint32_t k = 0; k < R; ++k) {
            for (uint32_t s = 0; s < NSEL; ++s) {
                matmul_tiles(cb_sel, cb_gath, s, (2 * k) * NSEL + s, k);
                matmul_tiles(cb_sel, cb_gath, s, (2 * k + 1) * NSEL + s, k);
            }
        }
        tile_regs_commit();
        cb_reserve_back(cb_tout, R);
        tile_regs_wait();
        pack_to(0, cb_tout);
        pack_tile(1, cb_tout);
        tile_regs_release();
        cb_push_back(cb_tout, R);
        cb_pop_front(cb_gath, GT);
    }
    // g = bf16(sum) (the Linear's bf16 output) via the packer
    tile_regs_acquire();
    cp_prep(cb_tot2);  // (configured before the totals arrive)
    cb_wait_front(cb_tot2, R);
    cp(cb_tot2, 0, 0);
    copy_tile(cb_tot2, 1, 1);
    tile_regs_commit();
    cb_reserve_back(cb_g, R);
    tile_regs_wait();
    pack_to(0, cb_g);
    pack_tile(1, cb_g);
    tile_regs_release();
    cb_push_back(cb_g, R);
    cb_pop_front(cb_tot2, R);
    cb_wait_front(cb_g, R);

    // ---- forget / gain on this core's tiles: tanh(g @ W)
    cb_wait_front(cb_wf, R * PER);
    cb_wait_front(cb_wg, R * PER);
    cb_reserve_back(cb_fa, 2 * PER);
    tile_regs_acquire();
    mm_begin(cb_g, cb_wf);
    for (uint32_t t = 0; t < PER; ++t) {
        for (uint32_t k = 0; k < R; ++k) {
            matmul_tiles(cb_g, cb_wf, k, k * PER + t, t);
            matmul_tiles(cb_g, cb_wg, k, k * PER + t, 4 + t);
        }
    }
    drain();
    tanh_tile_init();
    for (uint32_t t = 0; t < PER; ++t) {
        rtanh(t);
        rtanh(4 + t);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_cfg(cb_fa);
    for (uint32_t t = 0; t < PER; ++t) {
        pack_tile(t, cb_fa);
    }
    for (uint32_t t = 0; t < PER; ++t) {
        pack_tile(4 + t, cb_fa);
    }
    tile_regs_release();
    cb_push_back(cb_fa, 2 * PER);
    cb_wait_front(cb_fa, 2 * PER);
    cb_pop_front(cb_g, R);
    cb_pop_front(cb_wf, R * PER);
    cb_pop_front(cb_wg, R * PER);

    // ---- out = h - f h + a m (NORM: kept as fp32 tiles + the round-3 partials sum out^2, hi / lo)
    if constexpr (NORM) {
        cb_reserve_back(cb_o32, PER);
        cb_reserve_back(cb_sq, 2 * PER);
    }
    for (uint32_t t = 0; t < PER; ++t) {
        tile_regs_acquire();
        cp(cb_fa, t, 0);
        copy_tile(cb_fa, PER + t, 1);
        copy_tile(cb_h, t, 2);
        copy_tile(cb_m, t, 3);
        sinit();
        mul(0, 2, 0);
        mul(1, 3, 1);
        sub(2, 0, 2);
        add(2, 1, 2);
        if constexpr (NORM) {
            mul(2, 2, 4);
            split(4, 4, 5);
            tile_regs_commit();
            tile_regs_wait();
            pack_to(2, cb_o32);
            pack_tile(4, cb_sq);
            pack_tile(5, cb_sq);
            tile_regs_release();
        } else {
            tile_regs_commit();
            cb_reserve_back(cb_out, 1);
            tile_regs_wait();
            pack_to(2, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, 1);
        }
    }
    cb_pop_front(cb_fa, 2 * PER);
    cb_pop_front(cb_h, PER);
    cb_pop_front(cb_m, PER);
    if constexpr (NORM) {
        cb_push_back(cb_o32, PER);
        cb_push_back(cb_sq, 2 * PER);
        cb_wait_front(cb_o32, PER);
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
        if (root) {
            tile_regs_acquire();
            mm_begin(cb_sel, cb_gath);  // (configured before the gather arrives)
            cb_wait_front(cb_gath, GT);
            for (uint32_t s = 0; s < NSEL; ++s) {
                matmul_tiles(cb_sel, cb_gath, s, s, 0);
                matmul_tiles(cb_sel, cb_gath, s, NSEL + s, 0);
            }
            tile_regs_commit();
            cb_reserve_back(cb_tout, 1);
            tile_regs_wait();
            pack_to(0, cb_tout);
            tile_regs_release();
            cb_push_back(cb_tout, 1);
            cb_pop_front(cb_gath, GT);
        }
        tile_regs_acquire();
        cp_prep(cb_tot3);  // (configured before the totals arrive)
        cb_wait_front(cb_tot3, 1);
        cp(cb_tot3, 0, 0);
        rnorm(0, inv_h, eps);
        tile_regs_commit();
        cb_reserve_back(cb_r, 1);
        tile_regs_wait();
        pack_to(0, cb_r);
        tile_regs_release();
        cb_push_back(cb_r, 1);
        cb_pop_front(cb_tot3, 1);
        cb_wait_front(cb_r, 1);
        cb_wait_front(cb_nw, PER);
        if constexpr (ROUTER) {
            cb_reserve_back(cb_xb, PER);
        }
        for (uint32_t i = 0; i < PER; ++i) {
            tile_regs_acquire();
            cp(cb_o32, i, 0);
            cp(cb_r, 0, 1);
            sinit();
            mul(0, 1, 0);
            cp(cb_nw, i, 1);
            sinit();
            mul(0, 1, 0);
            tile_regs_commit();
            cb_reserve_back(cb_out, 1);
            tile_regs_wait();
            pack_to(0, cb_out);
            if constexpr (ROUTER) {
                pack_tile(0, cb_xb);
            }
            tile_regs_release();
            cb_push_back(cb_out, 1);
        }
        cb_pop_front(cb_r, 1);
        cb_pop_front(cb_o32, PER);
        cb_pop_front(cb_nw, PER);
        if constexpr (ROUTER) {
            // round 4 partials: x (bf16) @ [W hi | W lo] rows of this core's tiles -> ET tiles -> hi / lo parts
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
            cb_reserve_back(cb_part, 2 * ET);
            tile_regs_wait();
            pack_cfg(cb_part);
            for (uint32_t e = 0; e < ET; ++e) {
                pack_tile(e, cb_part);
                pack_tile(4 + e, cb_part);
            }
            tile_regs_release();
            cb_push_back(cb_part, 2 * ET);
            cb_pop_front(cb_xb, PER);
            cb_pop_front(cb_rw, 2 * PER * ET);
#if G2_RSPLIT  // (core t ET + e reduces logit tile e of row t)
            const uint32_t ci = get_arg_val<uint32_t>(0);
            if (ci < ROWS * ET) {
                const uint32_t e = ci % ET;
                cb_wait_front(cb_sel, NSEL);
                tile_regs_acquire();
                mm_begin(cb_sel, cb_gath);  // (configured before the gather arrives)
                cb_wait_front(cb_gath, GT);
                for (uint32_t s = 0; s < NSEL; ++s) {
                    matmul_tiles(cb_sel, cb_gath, s, (2 * e) * NSEL + s, 0);
                    matmul_tiles(cb_sel, cb_gath, s, (2 * e + 1) * NSEL + s, 0);
                }
                tile_regs_commit();
                cb_reserve_back(cb_tout, 1);
                tile_regs_wait();
                pack_cfg(cb_tout);
                pack_tile(0, cb_tout);
                tile_regs_release();
                cb_push_back(cb_tout, 1);
                cb_pop_front(cb_gath, GT);
                if (!root) {
                    cb_pop_front(cb_sel, NSEL);
                }
            }
            if (false) {
#else
            if (root) {
#endif
                tile_regs_acquire();
                mm_begin(cb_sel, cb_gath);  // (configured before the gather arrives)
                cb_wait_front(cb_gath, GT);
                for (uint32_t e = 0; e < ET; ++e) {
                    for (uint32_t s = 0; s < NSEL; ++s) {
                        matmul_tiles(cb_sel, cb_gath, s, (2 * e) * NSEL + s, e);
                        matmul_tiles(cb_sel, cb_gath, s, (2 * e + 1) * NSEL + s, e);
                    }
                }
                tile_regs_commit();
                cb_reserve_back(cb_tout, ET);
                tile_regs_wait();
                pack_cfg(cb_tout);
                for (uint32_t e = 0; e < ET; ++e) {
                    pack_tile(e, cb_tout);
                }
                tile_regs_release();
                cb_push_back(cb_tout, ET);
                cb_pop_front(cb_gath, GT);
            }
        }
    }
    cb_pop_front(cb_ones, 1);
    if (root) {
        cb_pop_front(cb_sel, NSEL);
    }
}
