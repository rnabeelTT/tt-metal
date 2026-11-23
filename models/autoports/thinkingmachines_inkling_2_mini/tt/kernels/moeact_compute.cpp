// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Prefill MoE activation (tt/moeact.py), compute, fp32 (the op chain's steps and roundings):
//   a = bf16(tanh(g / 4) sigmoid(g) tanh(u))      (tt/moe.py inkling_act_fused; u carries the 1/25 of the weights)
//   out = bf16(a w_row)                           (the routing weight; the packer's bf16 conversion)
//
// runtime args: n (output tiles of this core)

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

#ifdef TRISC_MATH
namespace ckernel::sfpu {
template <int MODE, int ITERATIONS>
inline void calculate_ma_bin(const uint32_t a, const uint32_t b, const uint32_t o) {
    constexpr uint32_t tile = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[a * tile];
        sfpi::vFloat y = sfpi::dst_reg[b * tile];
        sfpi::vFloat p = x * y;
        if constexpr (MODE == 1) {  // bf16, round to nearest even (the typecast's)
            sfpi::vUInt u = sfpi::as<sfpi::vUInt>(p);
            sfpi::vUInt lsb = (u >> 16) & 1;
            u = (u + 0x7FFF + lsb) & 0xFFFF0000;
            p = sfpi::as<sfpi::vFloat>(u);
        }
        sfpi::dst_reg[o * tile] = p;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

namespace {
template <int MODE>
ALWI void bmul(uint32_t a, uint32_t b, uint32_t o) {
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_ma_bin, (MODE, 8), a, b, o, VectorMode::RC)));
}
ALWI void drain() {
#ifdef TRISC_MATH
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
#endif
}
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_g = get_compile_time_arg_val(0);
    constexpr uint32_t cb_u = get_compile_time_arg_val(1);
    constexpr uint32_t cb_w = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t Wt = get_compile_time_arg_val(4);
    constexpr uint32_t quarter = get_compile_time_arg_val(5);  // 0.25f bits
    const uint32_t nr = get_arg_val<uint32_t>(0);              // row tiles of this core
    compute_kernel_hw_startup(cb_g, cb_out);
    for (uint32_t r = 0; r < nr; ++r) {
        cb_wait_front(cb_w, 1);
        for (uint32_t c = 0; c < Wt; ++c) {
            cb_wait_front(cb_g, 1);
            cb_wait_front(cb_u, 1);
            tile_regs_acquire();
            // DST 0 g / 4 -> tanh, 1 g -> sigmoid, 2 u -> tanh, 3 the row weights
            copy_init(cb_g);
            copy_tile(cb_g, 0, 0);
            copy_tile(cb_g, 0, 1);
            copy_init(cb_u);
            copy_tile(cb_u, 0, 2);
            copy_init(cb_w);
            copy_tile(cb_w, 0, 3);
            drain();
            binop_with_scalar_tile_init();
            mul_unary_tile(0, quarter);
            tanh_tile_init();
            tanh_tile(0);
            tanh_tile(2);
            sigmoid_tile_init<false>();
            sigmoid_tile<VectorMode::RC, false>(1);
            drain();
            MATH((SFPU_BINARY_INIT(unused)));
            bmul<0>(0, 1, 0);
            bmul<1>(0, 2, 0);  // a = bf16(tanh(g / 4) sigmoid(g) tanh(u))
            bmul<0>(0, 3, 0);  // a w (fp32; the packer rounds to bf16)
            tile_regs_commit();
            cb_reserve_back(cb_out, 1);
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, 1);
            cb_pop_front(cb_g, 1);
            cb_pop_front(cb_u, 1);
        }
        cb_pop_front(cb_w, 1);
    }
}
