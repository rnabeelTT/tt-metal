// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Multi-row expert gate|up matmul + SwiGLU (TRISC) of the fused MoE decode (moe_decode/union.py ExpertGateUpRows),
// M = ROWS rows ([M, 32] activation tiles). Per union slot this core owns (count from BRISC through the mailboxes):
//   u = x . W[:, up col j], g = x . W[:, gate col j]          (custom_mm, LoFi, FP32 accumulation)
//   w = the slot's [M, 32] weight tile . I                   (custom_mm with the identity tile: loads it into DST)
//   act = w * silu(g) * u, rounded to 7 significant bits (the down projection's srcB width), one [M, 32] tile.
// The per-row weight load through an identity matmul follows the gpt-oss union kernels
// (models/demos/gpt_oss/tt/fused_decode/kernels/stream_gate_up_union_compute.cpp).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/pack.h"

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
        sfpi::vInt v = sfpi::reinterpret<sfpi::vInt>(x);
        v = (v + 0x10000) & sfpi::vInt(0xFFFE0000);
        sfpi::dst_reg[0] = sfpi::reinterpret<sfpi::vFloat>(v);
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

    constexpr bool transpose = false;
    constexpr bool split_acc = true;
    constexpr bool dense_packing = false;
    custom_mm_block_init<transpose, split_acc, dense_packing>(cb_x, cb_w, cb_act);

    uint32_t n = 0;
    UNPACK(n = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    MATH(n = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    PACK(n = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));

    cb_wait_front(cb_x, kt);
    cb_wait_front(cb_id, 2);
    for (uint32_t i = 0; i < n; ++i) {
        tile_regs_acquire();
        cb_wait_front(cb_w, kt);
        custom_mm_block<true>(cb_x, cb_w, 0, 0, 0, kt);  // up
        cb_pop_front(cb_w, kt);
        cb_wait_front(cb_w, kt);
        custom_mm_block<true>(cb_x, cb_w, 0, 0, 1, kt);  // gate
        cb_pop_front(cb_w, kt);
        cb_wait_front(cb_sc, 2);
        custom_mm_block<true>(cb_sc, cb_id, 0, 0, 2, 2);  // per-row weights
        cb_pop_front(cb_sc, 2);

        silu_tile_init();
        silu_rows(1);
        mul_binary_tile_init();
        mul_rows(0, 1, 0);
        mul_rows(0, 2, 0);
        round_rows(0);
        tile_regs_commit();

        cb_reserve_back(cb_act, 1);
        tile_regs_wait();
        pack_tile(0, cb_act);
        tile_regs_release();
        cb_push_back(cb_act, 1);

        custom_mm_block_init_short<transpose, split_acc, dense_packing>(cb_x, cb_w, cb_act);
    }
    cb_pop_front(cb_id, 2);
    cb_pop_front(cb_x, kt);
    custom_mm_block_uninit<dense_packing>();
}
