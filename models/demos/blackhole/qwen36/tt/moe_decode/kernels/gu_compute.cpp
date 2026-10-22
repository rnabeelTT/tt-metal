// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Expert gate|up matmul + SwiGLU (TRISC) of the fused MoE decode (moe_decode/experts.py ExpertGateUp).
//
// Per slot this core owns (shared expert or a routed expert) and its column pair j:
//   u = x . W[:, up col j],  g = x . W[:, gate col j]   (custom_mm, 1x32 activation tiles, LoFi, FP32 accumulation)
//   act = w_s * silu(g) * u                               (SFPU on the result rows; w_s from BRISC via the mailbox)
// rounded to the 7 significant bits the down projection's LoFi custom_mm reads exactly, packed as one 1x32 tile.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/pack.h"

// SFPU ops over the result rows only: a 1x32 matmul result occupies the first rows of DST faces 0 and 1
// (VectorMode::R, 2 iterations), as in the gpt-oss fused decode (models/demos/gpt_oss/tt/fused_decode/kernels).
constexpr uint32_t kRowIters = 2;
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
// Round to 7 significant bits (6 explicit mantissa bits), half away from zero: the srcB operand width custom_mm
// multiplies at LoFi, so the down projection reads its activations exactly instead of truncated.
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
    constexpr uint32_t kt = get_compile_time_arg_val(3);

    constexpr bool transpose = false;
    constexpr bool split_acc = true;
    constexpr bool dense_packing = false;
    custom_mm_block_init<transpose, split_acc, dense_packing>(cb_x, cb_w, cb_act);

    const uint32_t num_slots = get_arg_val<uint32_t>(0);  // this core's slot count

    cb_wait_front(cb_x, kt);
    for (uint32_t i = 0; i < num_slots; ++i) {
        tile_regs_acquire();
        cb_wait_front(cb_w, kt);
        custom_mm_block<true>(cb_x, cb_w, 0, 0, 0, kt);  // up
        cb_pop_front(cb_w, kt);
        cb_wait_front(cb_w, kt);
        custom_mm_block<true>(cb_x, cb_w, 0, 0, 1, kt);  // gate
        cb_pop_front(cb_w, kt);

        silu_tile_init();
        silu_rows(1);
        mul_binary_tile_init();
        mul_rows(0, 1, 0);
        binop_with_scalar_tile_init();
        // the slot weight arrives once the router tail is done; the shared expert's matmuls run before that
        uint32_t score = 0;
        MATH(score = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
        scalar_rows<MUL_UNARY>(0, score);
        round_rows(0);
        tile_regs_commit();

        cb_reserve_back(cb_act, 1);
        tile_regs_wait();
        pack_tile(0, cb_act);
        tile_regs_release();
        cb_push_back(cb_act, 1);

        custom_mm_block_init_short<transpose, split_acc, dense_packing>(cb_x, cb_w, cb_act);
    }
    cb_pop_front(cb_x, kt);
    custom_mm_block_uninit<dense_packing>();
}
