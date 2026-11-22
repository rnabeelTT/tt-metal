// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Decode attention prep (tt/aprep.py), compute. Per unit (head-major tiles from the reader: row h = head h):
//   q, r, k: x * min(1, clip / ||x_h||) (* tau_b for q / r on global layers)        (L2 clip, transformers' l2_clip)
//   v:       x * rsqrt(mean(x_h^2) + eps)                                           (unweighted RMS)
//   sink:    copy
// -> bf16. The per-head sum of squares: (x * x) @ ones (FPU, TF32 operands: ~5e-4 relative, the outputs are bf16) puts
// each head's sum in every column of its row, so the scale is an elementwise multiply (SFPU, FP32 DST).

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

#ifdef TRISC_MATH
namespace ckernel::sfpu {
// DST[hi] = DST[x] with the low mantissa bits cleared (exact as a TF32 matmul operand), DST[lo] = DST[x] - DST[hi]
// (qwen36 gdn_decode's split): the row sums of x^2 then carry ~2^-22 instead of TF32's ~2^-11 truncation
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
// DST = min(DST * thr, 1) * tau (the L2 clip factor from rsqrt(sum x^2))
template <int ITERATIONS = 8>
inline void calculate_aprep_clip(const uint32_t thr_bits, const uint32_t tau_bits) {
    union {
        uint32_t u;
        float f;
    } t{thr_bits}, g{tau_bits};
    const sfpi::vFloat thr = t.f, tau = g.f;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0] * thr;
        v_if(v > 1.0f) { v = 1.0f; }
        v_endif;
        sfpi::dst_reg[0] = v * tau;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

namespace {
constexpr uint32_t IN0_MASK = 0xFFFFE000u;  // matmul in0 (srcB) keeps this many mantissa bits
ALWI void split_x2(uint32_t x, uint32_t hi, uint32_t lo) {
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_aprep_split, (IN0_MASK, 8), x, hi, lo, VectorMode::RC)));
}
ALWI void clip_scale(uint32_t idst, uint32_t thr, uint32_t tau) {
    MATH((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_aprep_clip, (8), idst, VectorMode::RC, thr, tau)));
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
    constexpr uint32_t RT_UNUSED = get_compile_time_arg_val(8);
    constexpr uint32_t GLOBAL = get_compile_time_arg_val(9);
    constexpr uint32_t qk_clip = get_compile_time_arg_val(10);   // fp32 bits
    constexpr uint32_t rel_clip = get_compile_time_arg_val(11);
    constexpr uint32_t inv_hd = get_compile_time_arg_val(12);
    constexpr uint32_t eps = get_compile_time_arg_val(13);
    constexpr uint32_t sink_scale = get_compile_time_arg_val(14);  // 1 / softmax scale (the SDPA's sink convention)
    constexpr uint32_t ONE = 0x3F800000u;
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t step = get_arg_val<uint32_t>(1);
    const uint32_t total = get_arg_val<uint32_t>(2);

    compute_kernel_hw_startup(cb_xf, cb_out);
    cb_wait_front(cb_ones, 1);
    for (uint32_t u = start; u < total; u += step) {
        const uint32_t type = u % 5;
        if (type == 4) {  // sink
            cb_wait_front(cb_xf, 1);
            cb_reserve_back(cb_out, 1);
            tile_regs_acquire();
            copy_from(cb_xf);
            copy_tile(cb_xf, 0, 0);
            binop_with_scalar_tile_init();
            mul_unary_tile(0, sink_scale);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_out);
            pack_tile(0, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, 1);
            cb_pop_front(cb_xf, 1);
            continue;
        }
        const uint32_t cb_x = type <= 1 ? cb_xf : cb_xb;
        const uint32_t nt = type == 1 ? 1 : HT;
        cb_wait_front(cb_x, nt);
        // x^2 -> cb_sq (fp32): hi parts (tiles 0..nt-1), lo parts (nt..2nt-1)
        cb_reserve_back(cb_sq, 2 * nt);
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
        cb_push_back(cb_sq, 2 * nt);
        // per-head scale f -> cb_f (fp32)
        cb_wait_front(cb_sq, 2 * nt);
        uint32_t tau = ONE;
        const bool use_tau = GLOBAL && type <= 1;
        if (use_tau) {
            cb_wait_front(cb_tau, 1);
            UNPACK((invalidate_l1_cache()));
            tau = read_tile_value(cb_tau, 0, 0);
        }
        cb_reserve_back(cb_f, 1);
        tile_regs_acquire();
        reconfig_data_format(cb_ones, cb_sq);  // matmul: in1 -> srcA, in0 -> srcB
        matmul_init(cb_sq, cb_ones, 0);
        for (uint32_t c = 0; c < 2 * nt; ++c) {
            matmul_tiles(cb_sq, cb_ones, c, 0, 0);  // row sums (hi + lo), in every column
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
            clip_scale(0, type == 1 ? rel_clip : qk_clip, tau);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_f);
        pack_tile(0, cb_f);
        tile_regs_release();
        cb_push_back(cb_f, 1);
        cb_pop_front(cb_sq, 2 * nt);
        if (use_tau) {
            cb_pop_front(cb_tau, 1);
        }
        // x * f -> bf16
        cb_wait_front(cb_f, 1);
        cb_reserve_back(cb_out, nt);
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
        pack_reconfig_data_format(cb_out);
        for (uint32_t c = 0; c < nt; ++c) {
            pack_tile(c, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, nt);
        cb_pop_front(cb_f, 1);
        cb_pop_front(cb_x, nt);
    }
    cb_pop_front(cb_ones, 1);
}
