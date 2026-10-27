// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Multi-row expert gate|up matmul + SwiGLU (TRISC) of the fused MoE decode (moe_decode/union.py ExpertGateUpRows),
// row groups of ROWS rows ([ROWS, 32] activation tiles). Per union slot this core owns (count from BRISC through the
// mailboxes) and row group:
//   u = x . W[:, up col], g = x . W[:, gate col]              (custom_mm over the group's bank blocks, ct = m; LoFi,
//                                                              FP32 accumulation; column pairs as gu_layout.hpp)
//   w = the slot's [M, 32] weight tile . I                   (custom_mm with the identity tile: loads it into DST)
//   act = w * silu(g) * u, rounded to 7 significant bits (the down projection's srcB width), one [M, 32] tile per pair.
// SPLIT_PAIR (m = 1, banks q / q + banks / 2 hold the up / gate column of pair q): each core streams one bank;
// ROLE_GATE cores pack g' = w * silu(g) (their writer sends it to the paired up core's cb_sc), the up cores load g'
// through the identity matmul like the weight tile and pack act = u * g', rounded.
// The per-row weight load through an identity matmul follows the gpt-oss union kernels
// (models/demos/gpt_oss/tt/fused_decode/kernels/stream_gate_up_union_compute.cpp).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/pack.h"
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
    constexpr uint32_t m = get_compile_time_arg_val(7);       // up|gate columns per bank and expert
    constexpr uint32_t banks = get_compile_time_arg_val(8);
#ifdef SPLIT_PAIR
    // split-pair (m = 1): one column per core; DST 0 = x . W[:, col], DST 1 = the [trows, 32] tile in cb_sc (the gate
    // core: the rows' routing weights; the up core: its partner's silu(g) * w)
    static_assert(m == 1, "split-pair: one column per bank");
    constexpr uint32_t nb = 1;
    constexpr uint32_t np = 1;
    constexpr uint32_t wdst = 1;
#else
    constexpr uint32_t nb = gu_layout::blocks<m>();
    constexpr uint32_t np = gu_layout::pairs<m>();
    constexpr uint32_t wdst = gu_layout::tiles<m>();  // the per-row weight tile, after the matmul results
#endif

    constexpr bool transpose = false;
    constexpr bool split_acc = true;
    constexpr bool dense_packing = false;
    constexpr uint32_t ct = m;  // one bank block's columns
    custom_mm_block_init<transpose, split_acc, dense_packing>(cb_x, cb_w, cb_act, ct);

    uint32_t n = 0;
    UNPACK(n = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    MATH(n = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    PACK(n = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));

    cb_wait_front(cb_x, groups * kt);
    cb_wait_front(cb_id, 2);
    for (uint32_t i = 0; i < n; ++i) {
        cb_wait_front(cb_w, nb * kt * m);  // the slot's bank blocks (consecutive in the ring: an even ring when nb = 2)
        cb_wait_front(cb_sc, 2 * groups);
        for (uint32_t g = 0; g < groups; ++g) {
            tile_regs_acquire();
            gu_layout::static_for<nb>([&](auto h) {
                constexpr uint32_t hb = decltype(h)::value;
                custom_mm_block<true>(cb_x, cb_w, g * kt, hb * kt * m, hb * m, kt, m);
            });
            custom_mm_block_init_short<transpose, split_acc, dense_packing>(cb_sc, cb_id, cb_act, 1);
            custom_mm_block<true>(cb_sc, cb_id, 2 * g, 0, wdst, 2);  // the group's per-row weights

#ifdef SPLIT_PAIR
#ifdef ROLE_GATE
            silu_tile_init();
            silu_rows(0);  // g' = silu(g) * w, sent to the up core
            mul_binary_tile_init();
            mul_rows(0, wdst, 0);
#else
            mul_binary_tile_init();
            mul_rows(0, wdst, 0);  // act = u * g'
            round_rows(0);
#endif
            tile_regs_commit();
            cb_reserve_back(cb_act, 1);
            tile_regs_wait();
            pack_tile(0, cb_act);
            tile_regs_release();
            cb_push_back(cb_act, 1);
            custom_mm_block_init_short<transpose, split_acc, dense_packing>(cb_x, cb_w, cb_act, ct);
            continue;
#endif
            silu_tile_init();
            gu_layout::static_for<np>([&](auto p) {
                constexpr uint32_t u = gu_layout::up_dst<m, banks>(decltype(p)::value);
#ifdef GATE_FIRST
                silu_rows(u);  // [gate | up] packing: the first-half column is the gate
#else
                silu_rows(gu_layout::gate_dst<m, banks>(u));
#endif
            });
            mul_binary_tile_init();
            gu_layout::static_for<np>([&](auto p) {
                constexpr uint32_t u = gu_layout::up_dst<m, banks>(decltype(p)::value);
                mul_rows(u, gu_layout::gate_dst<m, banks>(u), u);
                mul_rows(u, wdst, u);
            });
            gu_layout::static_for<np>([&](auto p) { round_rows(gu_layout::up_dst<m, banks>(decltype(p)::value)); });
            tile_regs_commit();

            cb_reserve_back(cb_act, np);
            tile_regs_wait();
            gu_layout::static_for<np>([&](auto p) { pack_tile(gu_layout::up_dst<m, banks>(decltype(p)::value), cb_act); });
            tile_regs_release();
            cb_push_back(cb_act, np);

            custom_mm_block_init_short<transpose, split_acc, dense_packing>(cb_x, cb_w, cb_act, ct);
        }
        cb_pop_front(cb_sc, 2 * groups);
        cb_pop_front(cb_w, nb * kt * m);
    }
    cb_pop_front(cb_id, 2);
    cb_pop_front(cb_x, groups * kt);
    custom_mm_block_uninit<dense_packing>();
}
