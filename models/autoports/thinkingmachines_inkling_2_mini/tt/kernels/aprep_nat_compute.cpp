// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Prefill attention prep in the projections' layout (tt/aprep.py NAT), compute: aprep_compute.cpp's per-row math on
// whole tiles whose rows are the chunk's tokens (one head per job):
//   q, r, k: x * min(1, clip / ||x||) (* tau per row on global layers: the reader's tau tile)
//   v:       x * rsqrt(mean(x^2) + eps)
//   sink:    x / scale
// -> bf16. The row sums: (x * x) hi / lo split @ ones (FPU), as aprep_compute.cpp; tau multiplies the clip factor
// (min(f * thr, 1) * tau, the same fp32 product as the per-row kernel's scalar tau).

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "aprep_nat.hpp"

#ifdef TRISC_MATH
namespace ckernel::sfpu {
template <uint32_t MASK, int ITERATIONS = 8>
inline void calculate_aprep_split(const uint32_t x, const uint32_t hi, const uint32_t lo) {
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
inline void calculate_aprep_mul(const uint32_t a, const uint32_t b, const uint32_t o) {
    constexpr uint32_t tile = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::dst_reg[o * tile] = sfpi::dst_reg[a * tile] * sfpi::dst_reg[b * tile];
        sfpi::dst_reg++;
    }
}
// DST = min(DST * thr, 1) [* DST[tau]] (the clip factor; tau per row from a tile)
template <bool TAU, int ITERATIONS = 8>
inline void calculate_nat_clip(const uint32_t thr_bits, const uint32_t tau_idx) {
    constexpr uint32_t tile = 32;
    union {
        uint32_t u;
        float f;
    } t{thr_bits};
    const sfpi::vFloat thr = t.f;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0] * thr;
        v_if(v > 1.0f) { v = 1.0f; }
        v_endif;
        if constexpr (TAU) {
            sfpi::dst_reg[0] = v * sfpi::dst_reg[tau_idx * tile];
        } else {
            sfpi::dst_reg[0] = v * 1.0f;
        }
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

namespace {
constexpr uint32_t IN0_MASK = 0xFFFFE000u;
ALWI void split_x2(uint32_t x, uint32_t hi, uint32_t lo) {
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_aprep_split, (IN0_MASK, 8), x, hi, lo, VectorMode::RC)));
}
ALWI void clip_tau(uint32_t thr, uint32_t tau_idx) {
    MATH((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_nat_clip, (true, 8), 0, VectorMode::RC, thr, tau_idx)));
}
ALWI void clip_one(uint32_t thr) {
    MATH((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_nat_clip, (false, 8), 0, VectorMode::RC, thr, 0)));
}
ALWI void copy_from(uint32_t cb) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_xf = get_compile_time_arg_val(0);
    constexpr uint32_t cb_xb = get_compile_time_arg_val(1);
    constexpr uint32_t cb_tau = get_compile_time_arg_val(2);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(3);
    constexpr uint32_t cb_sq = get_compile_time_arg_val(4);
    constexpr uint32_t cb_f = get_compile_time_arg_val(5);
    constexpr uint32_t cb_out = get_compile_time_arg_val(6);
    constexpr uint32_t HT = get_compile_time_arg_val(7);
    constexpr uint32_t GLOBAL = get_compile_time_arg_val(8);
    constexpr uint32_t qk_clip = get_compile_time_arg_val(9);
    constexpr uint32_t rel_clip = get_compile_time_arg_val(10);
    constexpr uint32_t inv_hd = get_compile_time_arg_val(11);
    constexpr uint32_t eps = get_compile_time_arg_val(12);
    constexpr uint32_t sink_scale = get_compile_time_arg_val(13);
    constexpr uint32_t cb_out1 = get_compile_time_arg_val(14);  // (1-tile outputs: r heads, the sink)
    constexpr uint32_t cb_x1 = get_compile_time_arg_val(15);    // (1-tile fp32 inputs: r heads, the sink)
    // (every CB gets pushes of one size only: multi-tile accesses never wrap around a CB's end)
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t step = get_arg_val<uint32_t>(1);
    const uint32_t total = get_arg_val<uint32_t>(2);

    compute_kernel_hw_startup(cb_xf, cb_out);
    cb_reserve_back(cb_ones, 1);
    tile_regs_acquire();
    fill_tile_init();
    fill_tile(0, 1.0f);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_ones);
    pack_tile(0, cb_ones);
    tile_regs_release();
    cb_push_back(cb_ones, 1);
    cb_wait_front(cb_ones, 1);

    for (uint32_t u = start; u < total; u += step) {
        uint32_t type, h0, h1;
        nat_unit(u % NAT_NU, type, h0, h1);
        if (type == 4) {  // sink
            cb_wait_front(cb_x1, 1);
            cb_reserve_back(cb_out1, 1);
            tile_regs_acquire();
            copy_from(cb_x1);
            copy_tile(cb_x1, 0, 0);
            binop_with_scalar_tile_init();
            mul_unary_tile(0, sink_scale);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_out1);
            pack_tile(0, cb_out1);
            tile_regs_release();
            cb_push_back(cb_out1, 1);
            cb_pop_front(cb_x1, 1);
            continue;
        }
        const bool use_tau = GLOBAL && type <= 1;
        if (use_tau) {
            cb_wait_front(cb_tau, 1);
        }
        const uint32_t cb_x = type == 0 ? cb_xf : type == 1 ? cb_x1 : cb_xb;
        const uint32_t nt = type == 1 ? 1 : HT;
        const uint32_t nh = type <= 1 ? h1 - h0 : 1;
        for (uint32_t j = 0; j < nh; ++j) {
            cb_wait_front(cb_x, nt);
            cb_reserve_back(cb_sq, 2 * HT);
            tile_regs_acquire();
            copy_from(cb_x);
            for (uint32_t c = 0; c < nt; ++c) {
                copy_tile(cb_x, c, c);
            }
            mul_binary_tile_init();
            for (uint32_t c = 0; c < nt; ++c) {
                mul_binary_tile(c, c, c);
            }
            MATH((SFPU_BINARY_INIT(unused)));
            for (uint32_t c = 0; c < nt; ++c) {
                split_x2(c, c, 4 + c);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_sq);
            for (uint32_t c = 0; c < nt; ++c) {
                pack_tile(c, cb_sq);
            }
            for (uint32_t c = 0; c < nt; ++c) {
                pack_tile(4 + c, cb_sq);
            }
            tile_regs_release();
            cb_push_back(cb_sq, 2 * HT);
            cb_wait_front(cb_sq, 2 * HT);
            cb_reserve_back(cb_f, 1);
            tile_regs_acquire();
            if (use_tau) {  // (tau into DST 1 first: the matmul accumulates into DST 0)
                copy_from(cb_tau);
                copy_tile(cb_tau, 0, 1);
            }
            reconfig_data_format(cb_ones, cb_sq);
            matmul_init(cb_sq, cb_ones, 0);
            for (uint32_t c = 0; c < 2 * nt; ++c) {
                matmul_tiles(cb_sq, cb_ones, c, 0, 0);
            }
            if (type == 3) {
                binop_with_scalar_tile_init();
                mul_unary_tile(0, inv_hd);
                add_unary_tile(0, eps);
                rsqrt_tile_init<false>();
                rsqrt_tile<false>(0);
            } else {
                rsqrt_tile_init<false>();
                rsqrt_tile<false>(0);
                binop_with_scalar_tile_init();
                if (use_tau) {
                    clip_tau(type == 1 ? rel_clip : qk_clip, 1);
                } else {
                    clip_one(type == 1 ? rel_clip : qk_clip);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_f);
            pack_tile(0, cb_f);
            tile_regs_release();
            cb_push_back(cb_f, 1);
            cb_pop_front(cb_sq, 2 * HT);
            const uint32_t cbo = nt == 1 ? cb_out1 : cb_out;
            cb_wait_front(cb_f, 1);
            cb_reserve_back(cbo, nt);
            tile_regs_acquire();
            copy_from(cb_f);
            copy_tile(cb_f, 0, 7);
            copy_from(cb_x);
            for (uint32_t c = 0; c < nt; ++c) {
                copy_tile(cb_x, c, c);
            }
            mul_binary_tile_init();
            for (uint32_t c = 0; c < nt; ++c) {
                mul_binary_tile(c, 7, c);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cbo);
            for (uint32_t c = 0; c < nt; ++c) {
                pack_tile(c, cbo);
            }
            tile_regs_release();
            cb_push_back(cbo, nt);
            cb_pop_front(cb_f, 1);
            cb_pop_front(cb_x, nt);
        }
        if (use_tau) {
            cb_pop_front(cb_tau, 1);
        }
    }
    cb_pop_front(cb_ones, 1);
}
