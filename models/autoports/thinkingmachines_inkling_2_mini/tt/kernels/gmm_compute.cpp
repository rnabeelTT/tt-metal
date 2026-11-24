// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Grouped expert matmul (tt/gmm.py), compute: per group, per row tile: [1 x Kt] @ [Kt x nb] accumulated in DST over
// the whole K (the program's fidelity, FP32 DST), packed to nb output tiles.
//
// runtime args: e0, e1 (this core's experts), G. The weight block stays at the CB front for a run of groups of one
// expert (the ids from cb_ids, sorted); padding groups (id >= E) end the scan.
// GMM_ACT: DST j < h holds the gate, h + j the up products (h = nb / 2); each is rounded to bf16 (the packer's
// rounding of the unfused gate|up output) and the h outputs are tt/kernels/moeact_compute.cpp's math:
// bf16(bf16(tanh(g / 4) sigmoid(g) tanh(u)) w).

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#if GMM_ACT
#include "api/compute/common.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/compute_kernel_api.h"

#ifdef TRISC_MATH
namespace ckernel::sfpu {
// MODE 0: o = a b; 1: o = bf16(a b) (the typecast's rounding); 2: o = bf16(a) (the packer's; b unused)
template <int MODE, int ITERATIONS>
inline void calculate_ga_bin(const uint32_t a, const uint32_t b, const uint32_t o) {
    constexpr uint32_t tile = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat p = sfpi::dst_reg[a * tile];
        if constexpr (MODE != 2) {
            p = p * sfpi::dst_reg[b * tile];
        }
        if constexpr (MODE == 1) {  // (round to nearest even: the typecast's)
            sfpi::vUInt u = sfpi::as<sfpi::vUInt>(p);
            sfpi::vUInt lsb = (u >> 16) & 1;
            u = (u + 0x7FFF + lsb) & 0xFFFF0000;
            p = sfpi::as<sfpi::vFloat>(u);
        } else if constexpr (MODE == 2) {  // (ties away from zero: the packer's fp32 -> bf16)
            sfpi::vUInt u = sfpi::as<sfpi::vUInt>(p);
            u = (u + 0x8000) & 0xFFFF0000;
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
ALWI void gbin(uint32_t a, uint32_t b, uint32_t o) {
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_ga_bin, (MODE, 8), a, b, o, VectorMode::RC)));
}
ALWI void drain() {
#ifdef TRISC_MATH
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
#endif
}
}  // namespace
#endif

void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t Kt = get_compile_time_arg_val(3);
    constexpr uint32_t nb = get_compile_time_arg_val(4);
    constexpr uint32_t mt = get_compile_time_arg_val(5);
    constexpr uint32_t cb_ids = get_compile_time_arg_val(6);
    constexpr uint32_t E = get_compile_time_arg_val(7);
    const uint32_t e0 = get_arg_val<uint32_t>(0), e1 = get_arg_val<uint32_t>(1), G = get_arg_val<uint32_t>(2);
#if GMM_ACT
    constexpr uint32_t cb_rw = get_compile_time_arg_val(8);
    constexpr uint32_t quarter = get_compile_time_arg_val(9);  // 0.25f bits
    constexpr uint32_t h = nb / 2, dw = 2 * h, dt = 2 * h + 1;  // (DST: the weights, a scratch tile)
#endif
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_w, cb_out);  // (matmul: in1 -> srcA, the stock bmm's order)
    matmul_block_init(cb_a, cb_w, false, nb, 1, Kt);
#if GMM_RM
    pack_untilize_dest_init<nb>(cb_out);  // (row-major output: 32 rows x nb tiles per block)
#endif
    cb_wait_front(cb_ids, 1);
    uint32_t prev = 0xFFFFFFFF;
    for (uint32_t g = 0; g < G; ++g) {
        const uint32_t e = (read_tile_value(cb_ids, 0, g >> 1) >> ((g & 1) * 16)) & 0xFFFF;
        if (e < e0) {
            continue;
        }
        if (e >= e1) {
            break;
        }
        if (e != prev) {
            if (prev != 0xFFFFFFFF) {
                cb_pop_front(cb_w, Kt * nb);
            }
            cb_wait_front(cb_w, Kt * nb);
            prev = e;
        }
        for (uint32_t mi = 0; mi < mt; ++mi) {
            cb_wait_front(cb_a, Kt);
            tile_regs_acquire();
#if GMM_ACT
            matmul_block_init(cb_a, cb_w, false, nb, 1, Kt);
#endif
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                matmul_block(cb_a, cb_w, kt, kt * nb, 0, false, nb, 1, Kt);
            }
#if GMM_ACT
            cb_wait_front(cb_rw, 1);
            reconfig_data_format_srca(cb_w, cb_rw);  // (matmul: in1 = the weights in srcA)
            copy_init(cb_rw);
            copy_tile(cb_rw, 0, dw);
            drain();
            MATH((SFPU_BINARY_INIT(unused)));
            for (uint32_t j = 0; j < 2 * h; ++j) {
                gbin<2>(j, j, j);  // (the bf16 gate|up products)
            }
            for (uint32_t j = 0; j < h; ++j) {
                gbin<2>(j, j, dt);  // (g, for the sigmoid)
                drain();
                binop_with_scalar_tile_init();
                mul_unary_tile(j, quarter);
                tanh_tile_init();
                tanh_tile(j);
                tanh_tile(h + j);
                sigmoid_tile_init<false>();
                sigmoid_tile<VectorMode::RC, false>(dt);
                drain();
                MATH((SFPU_BINARY_INIT(unused)));
                gbin<0>(j, dt, j);
                gbin<1>(j, h + j, j);
                gbin<0>(j, dw, j);
            }
            cb_pop_front(cb_rw, 1);
            reconfig_data_format_srca(cb_rw, cb_w);
            tile_regs_commit();
            cb_reserve_back(cb_out, h);
            tile_regs_wait();
            for (uint32_t j = 0; j < h; ++j) {
                pack_tile(j, cb_out);
            }
            tile_regs_release();
            cb_push_back(cb_out, h);
            cb_pop_front(cb_a, Kt);
            continue;
#endif
            tile_regs_commit();
            cb_reserve_back(cb_out, nb);
            tile_regs_wait();
#if GMM_RM
            pack_untilize_dest<nb>(cb_out);
#else
            for (uint32_t j = 0; j < nb; ++j) {
                pack_tile(j, cb_out);
            }
#endif
            tile_regs_release();
            cb_push_back(cb_out, nb);
            cb_pop_front(cb_a, Kt);
        }
    }
    if (prev != 0xFFFFFFFF) {
        cb_pop_front(cb_w, Kt * nb);
    }
}
