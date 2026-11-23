// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// One-bank multi-row expert gate|up matmul + SwiGLU (TRISC) of the fused MoE decode (moe_decode/union.py
// ExpertGateUpRows.onebank; m odd: a pair's up column sits in bank b, its gate column in bank b +- banks / 2). Per union
// slot this core owns and row group:
//   S1: x . W[:, its bank's m columns] (custom_mm, ct = m, LoFi, FP32 DST as gur_compute.cpp), packed as full 32 x 32
//       FP32 tiles (the whole DST tile image, so the unpack restores it exactly):
//       the up columns (c < nup) to cb_up (kept), the gate columns to cb_xo (the writer sends them to the partner).
//   S2 (one slot later: the partner's tiles are in flight meanwhile): the up tiles and the partner's gate tiles (cb_xi,
//       in the same column order) back into DST unpacked as FP32 (exact: the values the two-bank kernel had in DST),
//       the per-row weight tile by the identity matmul, then the same SFPU sequence as gur_compute.cpp per pair (clamps,
//       silu(g), u * silu(g), * w, round to 7 significant bits) and one [M, 32] BF16 tile per pair to cb_act.
// ROLE_HI (banks b >= banks / 2): nup = (m - 1) / 2 up columns, else (m + 1) / 2.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "gu_layout.hpp"

// SFPU over the result rows only: M rows of DST faces 0 and 1 (VectorMode::R).
constexpr uint32_t kRowIters = ROWS <= 4 ? 2 : ROWS / 2;
ALWI void silu_rows(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_silu, (DST_ACCUM_MODE, kRowIters), idst, VectorMode::R));
}
ALWI void mul_rows(uint32_t a, uint32_t b, uint32_t out) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary_mul,
        (APPROX, ckernel::BinaryOp::MUL, kRowIters, DST_ACCUM_MODE),
        a,
        b,
        out,
        VectorMode::R)));
}

#ifdef TRISC_MATH
namespace ckernel::sfpu {
// Round to 7 significant bits (6 explicit mantissa bits), half away from zero.
template <int ITERATIONS>
inline void round_sig7() {
    for (int d = 0; d < ITERATIONS; d++) {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vInt v = sfpi::as<sfpi::vInt>(x);
        v = (v + 0x10000) & sfpi::vInt(0xFFFE0000);
        sfpi::dst_reg[0] = sfpi::as<sfpi::vFloat>(v);
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif


#ifdef SWIGLU_LIMIT
#ifdef TRISC_MATH
namespace ckernel::sfpu {
// SwiGLU input clamps (Hy4-flash: gate <= L, |up| <= L), on the result rows.
template <int ITERATIONS, bool BOTH>
inline void clamp_lim() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        v_if(x > SWIGLU_LIMIT) { x = SWIGLU_LIMIT; }
        v_endif;
        if constexpr (BOTH) {
            v_if(x < -SWIGLU_LIMIT) { x = -SWIGLU_LIMIT; }
            v_endif;
        }
        sfpi::dst_reg[0] = x;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif
ALWI void clamp_gate_rows(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, clamp_lim, (kRowIters, false), idst, VectorMode::R));
}
ALWI void clamp_up_rows(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, clamp_lim, (kRowIters, true), idst, VectorMode::R));
}
#define SWIGLU_CLAMP_GATE(i) clamp_gate_rows(i)
#define SWIGLU_CLAMP_UP(i) clamp_up_rows(i)
#else
#define SWIGLU_CLAMP_GATE(i)
#define SWIGLU_CLAMP_UP(i)
#endif

ALWI void round_rows(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, round_sig7, (kRowIters), idst, VectorMode::R));
}

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_act = get_compile_time_arg_val(2);
    constexpr uint32_t cb_sc = get_compile_time_arg_val(3);
    constexpr uint32_t cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t kt = get_compile_time_arg_val(5);
    constexpr uint32_t groups = get_compile_time_arg_val(6);  // row groups of ROWS rows
    constexpr uint32_t m = get_compile_time_arg_val(7);       // columns per bank and expert (odd)
    constexpr uint32_t cb_up = get_compile_time_arg_val(9);
    constexpr uint32_t cb_xo = get_compile_time_arg_val(10);
    constexpr uint32_t cb_xi = get_compile_time_arg_val(11);
#ifdef ROLE_HI
    constexpr uint32_t nup = (m - 1) / 2;
#else
    constexpr uint32_t nup = (m + 1) / 2;
#endif
    constexpr uint32_t nsend = m - nup;
    constexpr uint32_t nmax = (m + 1) / 2;  // cb_xi slot stride per row group
    constexpr uint32_t wdst = 2 * nup;      // the per-row weight tile: after the up and gate tiles

    constexpr bool transpose = false;
    constexpr CustomMmDestLayout mm_layout = CustomMmDestLayout::DefaultSplitAcc;
    custom_mm_block_init<transpose, mm_layout>(cb_x, cb_w, cb_up, m);

    uint32_t n = 0;
    UNPACK(n = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    MATH(n = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    PACK(n = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));

    cb_wait_front(cb_x, groups * kt);
    cb_wait_front(cb_id, 2);

    auto s1 = [&]() {
        cb_wait_front(cb_w, kt * m);
        reconfig_data_format_srca(cb_w);  // (matmul: srcA = in1, srcB = in0; the copies only use srcA)
        custom_mm_block_init_short<transpose, mm_layout>(cb_x, cb_w, cb_up, m);
        pack_reconfig_data_format(cb_up);
        pack_init(cb_up);  // full 32 x 32 FP32 tiles: the DST image round-trips unchanged (a [M, 32] tile's unpack to
                           // DST places face 1 elsewhere than custom_mm / the packer do)
        for (uint32_t g = 0; g < groups; ++g) {
            tile_regs_acquire();
            custom_mm_block<mm_layout, true>(cb_x, cb_w, g * kt, 0, 0, kt, m);
            tile_regs_commit();
            cb_reserve_back(cb_up, nup);
            cb_reserve_back(cb_xo, nsend);
            tile_regs_wait();
            gu_layout::static_for<nup>([&](auto c) { pack_tile(decltype(c)::value, cb_up); });
            gu_layout::static_for<nsend>([&](auto c) { pack_tile(nup + decltype(c)::value, cb_xo); });
            tile_regs_release();
            cb_push_back(cb_up, nup);
            cb_push_back(cb_xo, nsend);
        }
        cb_pop_front(cb_w, kt * m);
    };
    auto s2 = [&]() {
        cb_wait_front(cb_sc, 2 * groups);
        cb_wait_front(cb_xi, groups * nmax);
        pack_reconfig_data_format(cb_act);
        pack_init(cb_act);
        for (uint32_t g = 0; g < groups; ++g) {
            cb_wait_front(cb_up, nup);
            tile_regs_acquire();
            reconfig_data_format_srca(cb_up);
            copy_init(cb_up);
            gu_layout::static_for<nup>([&](auto c) { copy_tile(cb_up, decltype(c)::value, decltype(c)::value); });
            reconfig_data_format_srca(cb_xi);
            copy_init(cb_xi);
            gu_layout::static_for<nup>(
                [&](auto c) { copy_tile(cb_xi, g * nmax + decltype(c)::value, nup + decltype(c)::value); });
            reconfig_data_format_srca(cb_id);
            custom_mm_block_init_short<transpose, mm_layout>(cb_sc, cb_id, cb_act, 1);
            custom_mm_block<mm_layout, true>(cb_sc, cb_id, 2 * g, 0, wdst, 2);  // the group's per-row weights
            silu_tile_init();
            gu_layout::static_for<nup>([&](auto p) {
                constexpr uint32_t u = decltype(p)::value;
#ifdef GATE_FIRST
#error "one-bank layout: [up | gate] packing only"
#endif
                SWIGLU_CLAMP_GATE(nup + u);
                SWIGLU_CLAMP_UP(u);
                silu_rows(nup + u);
            });
            mul_binary_tile_init();
            gu_layout::static_for<nup>([&](auto p) {
                constexpr uint32_t u = decltype(p)::value;
                mul_rows(u, nup + u, u);
                mul_rows(u, wdst, u);
            });
            gu_layout::static_for<nup>([&](auto p) { round_rows(decltype(p)::value); });
            tile_regs_commit();
            cb_reserve_back(cb_act, nup);
            tile_regs_wait();
            gu_layout::static_for<nup>([&](auto p) { pack_tile(decltype(p)::value, cb_act); });
            tile_regs_release();
            cb_push_back(cb_act, nup);
            cb_pop_front(cb_up, nup);
        }
        cb_pop_front(cb_sc, 2 * groups);
        cb_pop_front(cb_xi, groups * nmax);
    };
    for (uint32_t i = 0; i < n; ++i) {
        s1();
        if (i > 0) {
            s2();
        }
    }
    if (n > 0) {
        s2();
    }
    cb_pop_front(cb_id, 2);
    cb_pop_front(cb_x, groups * kt);
    custom_mm_block_uninit<mm_layout>(cb_act);
}
