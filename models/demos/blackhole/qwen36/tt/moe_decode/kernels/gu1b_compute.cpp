// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// One-row expert gate|up matmul + SwiGLU (TRISC), one-bank layout (moe_decode/experts.py ExpertGateUp1B; m odd: a
// pair's up column sits in bank b, its gate column in bank b +- banks / 2). Per slot this core owns:
//   S1: x . W[:, its bank's m columns] (custom_mm, ct = m, 1x32 activation tiles, LoFi, FP32 DST: gu_compute.cpp's
//       matmul of that bank block), packed as full 32 x 32 FP32 tiles (the DST image round-trips exactly,
//       gub_compute.cpp): the up columns (c < nup) to cb_up (kept), the gate columns to cb_xo (to the partner).
//   S2 (one slot later: the partner's tiles are in flight meanwhile): the up tiles and the partner's gate tiles (cb_xi,
//       in up-column order) back into DST unpacked as FP32 (the values the two-bank kernel had), then gu_compute.cpp's
//       SFPU sequence per pair (clamps, silu(g), u * silu(g), * w_s (the slot weight from BRISC, mailbox), round to 7
//       significant bits) and one 1x32 BF16 tile per pair to cb_act (local up-column order).
// ROLE_HI (banks b >= banks / 2): nup = (m - 1) / 2 up columns, else (m + 1) / 2.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "gu_layout.hpp"

constexpr uint32_t kRowIters = 2;  // the 1x32 result rows of DST faces 0 and 1 (VectorMode::R), as gu_compute.cpp
template <int OP>
ALWI void scalar_rows(uint32_t idst, uint32_t s) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binop_with_scalar,
        (APPROX, OP, kRowIters, DST_ACCUM_MODE),
        idst,
        VectorMode::R,
        s));
}
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
    constexpr uint32_t kt = get_compile_time_arg_val(3);
    constexpr uint32_t m = get_compile_time_arg_val(4);
    constexpr uint32_t cb_up = get_compile_time_arg_val(5);
    constexpr uint32_t cb_xo = get_compile_time_arg_val(6);
    constexpr uint32_t cb_xi = get_compile_time_arg_val(7);
#ifdef ROLE_HI
    constexpr uint32_t nup = (m - 1) / 2;
#else
    constexpr uint32_t nup = (m + 1) / 2;
#endif
    constexpr uint32_t nsend = m - nup;
    constexpr uint32_t nmax = (m + 1) / 2;

    constexpr bool transpose = false;
    constexpr CustomMmDestLayout mm_layout = CustomMmDestLayout::DefaultSplitAcc;
    custom_mm_block_init<transpose, mm_layout>(cb_x, cb_w, cb_up, m);

    const uint32_t n = get_arg_val<uint32_t>(0);  // this core's slots
    cb_wait_front(cb_x, kt);

    auto s1 = [&]() {
        cb_wait_front(cb_w, kt * m);
        reconfig_data_format_srca(cb_w);  // (matmul: srcA = in1, srcB = in0; the copies use srcA)
        custom_mm_block_init_short<transpose, mm_layout>(cb_x, cb_w, cb_up, m);
        pack_reconfig_data_format(cb_up);
        pack_init(cb_up);
        tile_regs_acquire();
        custom_mm_block<mm_layout, true>(cb_x, cb_w, 0, 0, 0, kt, m);
        tile_regs_commit();
        cb_reserve_back(cb_up, nup);
        cb_reserve_back(cb_xo, nsend);
        tile_regs_wait();
        gu_layout::static_for<nup>([&](auto c) { pack_tile(decltype(c)::value, cb_up); });
        gu_layout::static_for<nsend>([&](auto c) { pack_tile(nup + decltype(c)::value, cb_xo); });
        tile_regs_release();
        cb_push_back(cb_up, nup);
        cb_push_back(cb_xo, nsend);
        cb_pop_front(cb_w, kt * m);
    };
    auto s2 = [&]() {
        cb_wait_front(cb_xi, nmax);
        cb_wait_front(cb_up, nup);
        pack_reconfig_data_format(cb_act);
        pack_init(cb_act);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_up);
        copy_init(cb_up);
        gu_layout::static_for<nup>([&](auto c) { copy_tile(cb_up, decltype(c)::value, decltype(c)::value); });
        reconfig_data_format_srca(cb_xi);
        copy_init(cb_xi);
        gu_layout::static_for<nup>([&](auto c) { copy_tile(cb_xi, decltype(c)::value, nup + decltype(c)::value); });
        silu_tile_init();
        gu_layout::static_for<nup>([&](auto p) {
            constexpr uint32_t u = decltype(p)::value;
            SWIGLU_CLAMP_GATE(nup + u);
            SWIGLU_CLAMP_UP(u);
            silu_rows(nup + u);
        });
        mul_binary_tile_init();
        gu_layout::static_for<nup>([&](auto p) {
            constexpr uint32_t u = decltype(p)::value;
            mul_rows(u, nup + u, u);
        });
        binop_with_scalar_tile_init();
        uint32_t score = 0;
        MATH(score = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
        gu_layout::static_for<nup>([&](auto p) { scalar_rows<MUL_UNARY>(decltype(p)::value, score); });
        gu_layout::static_for<nup>([&](auto p) { round_rows(decltype(p)::value); });
        tile_regs_commit();
        cb_reserve_back(cb_act, nup);
        tile_regs_wait();
        gu_layout::static_for<nup>([&](auto p) { pack_tile(decltype(p)::value, cb_act); });
        tile_regs_release();
        cb_push_back(cb_act, nup);
        cb_pop_front(cb_up, nup);
        cb_pop_front(cb_xi, nmax);
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
    cb_pop_front(cb_x, kt);
    custom_mm_block_uninit<mm_layout>(cb_act);
}
