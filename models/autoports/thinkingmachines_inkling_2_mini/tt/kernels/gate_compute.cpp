// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Message gate in one program (tt/gate.py), compute (transformers Inkling2LoopMessageGate):
//   r_h = rsqrt(mean(h^2) + eps), r_m likewise                                  (round 1, fp32: x^2 hi + lo @ ones)
//   g = bf16(Wd [bf16(h r_h) | bf16(m r_m)])                                   (round 2: bf16 operands, fp32 sums)
//   out = h - tanh(Wf g) h + tanh(Wg g) m                                       (local; fp32)
// Each core holds PER hidden tiles; the root (slot 0) also reduces the C partials of each round. DST: full sync, 8
// fp32 tiles.

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
#include "tools/profiler/kernel_profiler.hpp"

#ifdef TRISC_MATH
namespace ckernel::sfpu {
template <uint32_t MASK, int ITERATIONS = 8>
inline void calculate_gate_split(const uint32_t x, const uint32_t hi, const uint32_t lo) {
    constexpr uint32_t tile = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[x * tile];
        sfpi::vFloat h = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(v) & MASK);
        sfpi::dst_reg[hi * tile] = h;
        sfpi::dst_reg[lo * tile] = v - h;
        sfpi::dst_reg++;
    }
}
// DST[o] = DST[a] op DST[b] (0 add, 1 sub, 2 mul) over the first ITERATIONS SFPU rows of the selected faces
template <int OP, int ITERATIONS>
inline void calculate_gate_bin(const uint32_t a, const uint32_t b, const uint32_t o) {
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
// DST[o] = DST[a] + DST[b] over the first ITERATIONS SFPU rows of the faces the vector mode selects
template <int ITERATIONS>
inline void calculate_gate_add(const uint32_t a, const uint32_t b, const uint32_t o) {
    constexpr uint32_t tile = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[a * tile];
        sfpi::vFloat y = sfpi::dst_reg[b * tile];
        sfpi::dst_reg[o * tile] = x + y;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

#ifndef GATE_ROWS
#define GATE_ROWS 32
#endif
namespace {
// <= 16 live rows (decode): every SFPU pass on those rows only (VectorMode::R, 2 ceil(T / 4) iterations per face)
constexpr bool GROWS = GATE_ROWS <= 16;
constexpr int GIT = 2 * ((GATE_ROWS + 3) / 4);
constexpr uint32_t IN0_MASK = 0xFFFFE000u;
ALWI void split_x(uint32_t x, uint32_t hi, uint32_t lo) {
    if constexpr (GROWS) {
        MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_gate_split, (IN0_MASK, GIT), x, hi, lo, VectorMode::R)));
    } else {
        MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_gate_split, (IN0_MASK, 8), x, hi, lo, VectorMode::RC)));
    }
}
template <int OP>
ALWI void gbin(uint32_t a, uint32_t b, uint32_t o) {
    if constexpr (GROWS) {
        MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_gate_bin, (OP, GIT), a, b, o, VectorMode::R)));
    } else if constexpr (OP == 0) {
        add_binary_tile(a, b, o);
    } else if constexpr (OP == 1) {
        sub_binary_tile(a, b, o);
    } else {
        mul_binary_tile(a, b, o);
    }
}
template <int OP>
ALWI void gbin_init() {
    if constexpr (GROWS) {
        MATH((SFPU_BINARY_INIT(unused)));
    } else if constexpr (OP == 0) {
        add_binary_tile_init();
    } else if constexpr (OP == 1) {
        sub_binary_tile_init();
    } else {
        mul_binary_tile_init();
    }
}
ALWI void gtanh(uint32_t i) {
    if constexpr (GROWS) {
        MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_tanh, (false, DST_ACCUM_MODE, GIT), i, VectorMode::R));
    } else {
        tanh_tile(i);
    }
}
// FPU work (copy) after SFPU work on the same DST tiles in one acquire: the SFPU waits for the FPU, not the reverse
ALWI void sfpu_drain() {
#ifdef TRISC_MATH
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
#endif
}
// the root's partial sums: only the live rows (T <= 16: faces 0 / 1, 2 ceil(T / 4) SFPU iterations each: two per 4 rows; else every row)
ALWI void row_add(uint32_t a, uint32_t b, uint32_t o) {
    if constexpr (GATE_ROWS <= 16) {
        MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_gate_add, (2 * ((GATE_ROWS + 3) / 4)), a, b, o,
                               VectorMode::R)));
    } else {
        MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_gate_add, (8), a, b, o, VectorMode::RC)));
    }
}
ALWI void copy_from(uint32_t cb) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
}
ALWI void mm_begin(uint32_t in0, uint32_t in1) {
    reconfig_data_format(in1, in0);  // matmul: in1 -> srcA, in0 -> srcB
    matmul_init(in0, in1, 0);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_h = get_compile_time_arg_val(0);
    constexpr uint32_t cb_m = get_compile_time_arg_val(1);
    constexpr uint32_t cb_wd = get_compile_time_arg_val(2);
    constexpr uint32_t cb_wf = get_compile_time_arg_val(3);
    constexpr uint32_t cb_wg = get_compile_time_arg_val(4);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(5);
    constexpr uint32_t cb_sq = get_compile_time_arg_val(6);
    constexpr uint32_t cb_part1 = get_compile_time_arg_val(7);
    constexpr uint32_t cb_gath = get_compile_time_arg_val(8);
    constexpr uint32_t cb_res1 = get_compile_time_arg_val(9);
    constexpr uint32_t cb_r = get_compile_time_arg_val(10);
    constexpr uint32_t cb_hn = get_compile_time_arg_val(11);
    constexpr uint32_t cb_part2 = get_compile_time_arg_val(12);
    constexpr uint32_t cb_res2 = get_compile_time_arg_val(13);
    constexpr uint32_t cb_g = get_compile_time_arg_val(14);
    constexpr uint32_t cb_fa = get_compile_time_arg_val(15);
    constexpr uint32_t cb_out = get_compile_time_arg_val(16);
    constexpr uint32_t C = get_compile_time_arg_val(17);
    constexpr uint32_t PER = get_compile_time_arg_val(18);
    constexpr uint32_t R = get_compile_time_arg_val(19);
    constexpr uint32_t inv_h = get_compile_time_arg_val(20);  // fp32 bits of 1 / H
    constexpr uint32_t eps = get_compile_time_arg_val(21);
    static_assert(PER <= 3 && R <= 4, "DST budget");
    const bool root = get_arg_val<uint32_t>(0) == 0;

    compute_kernel_hw_startup(cb_h, cb_out);
    cb_wait_front(cb_ones, 1);
    cb_wait_front(cb_h, PER);
    cb_wait_front(cb_m, PER);

    {
    DeviceZoneScopedN("G_C_SQ");
    // ---- round 1: partial sums of h^2, m^2 (hi + lo parts, exact) -> cb_part1 [ss_h, ss_m]
    cb_reserve_back(cb_sq, 4 * PER);
    for (uint32_t which = 0; which < 2; ++which) {
        const uint32_t cbx = which ? cb_m : cb_h;
        tile_regs_acquire();
        copy_from(cbx);
        for (uint32_t t = 0; t < PER; ++t) {
            copy_tile(cbx, t, t);
        }
        gbin_init<2>();
        for (uint32_t t = 0; t < PER; ++t) {
            gbin<2>(t, t, t);
        }
        MATH((SFPU_BINARY_INIT(unused)));
        for (uint32_t t = 0; t < PER; ++t) {
            split_x(t, t, 4 + t);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_sq);
        for (uint32_t t = 0; t < PER; ++t) {
            pack_tile(t, cb_sq);
        }
        for (uint32_t t = 0; t < PER; ++t) {
            pack_tile(4 + t, cb_sq);
        }
        tile_regs_release();
    }
    cb_push_back(cb_sq, 4 * PER);
    cb_wait_front(cb_sq, 4 * PER);
    cb_reserve_back(cb_part1, 2);
    tile_regs_acquire();
    mm_begin(cb_sq, cb_ones);
    for (uint32_t i = 0; i < 2 * PER; ++i) {
        matmul_tiles(cb_sq, cb_ones, i, 0, 0);
    }
    for (uint32_t i = 2 * PER; i < 4 * PER; ++i) {
        matmul_tiles(cb_sq, cb_ones, i, 0, 1);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_part1);
    pack_tile(0, cb_part1);
    pack_tile(1, cb_part1);
    tile_regs_release();
    cb_push_back(cb_part1, 2);
    cb_pop_front(cb_sq, 4 * PER);

    }
    {
    DeviceZoneScopedN("G_C_ROOT1");
    // ---- root: reduce round 1 -> rsqrt factors
    if (root) {
        cb_wait_front(cb_gath, 2 * C);
        tile_regs_acquire();
        copy_from(cb_gath);
        copy_tile(cb_gath, 0, 0);
        copy_tile(cb_gath, 1, 1);
        MATH((SFPU_BINARY_INIT(unused)));
        for (uint32_t j = 1; j < C; ++j) {
            sfpu_drain();
            copy_from(cb_gath);
            copy_tile(cb_gath, 2 * j, 2);
            copy_tile(cb_gath, 2 * j + 1, 3);
            row_add(0, 2, 0);
            row_add(1, 3, 1);
        }
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv_h);
        mul_unary_tile(1, inv_h);
        add_unary_tile(0, eps);
        add_unary_tile(1, eps);
        rsqrt_tile_init<false>();
        rsqrt_tile<false>(0);
        rsqrt_tile<false>(1);
        tile_regs_commit();
        cb_pop_front(cb_gath, 2 * C);  // (before the result: the writer reuses cb_gath for round 2)
        cb_reserve_back(cb_res1, 2);
        tile_regs_wait();
        pack_reconfig_data_format(cb_res1);
        pack_tile(0, cb_res1);
        pack_tile(1, cb_res1);
        tile_regs_release();
        cb_push_back(cb_res1, 2);
    }

    }
    {
    DeviceZoneScopedN("G_C_NORM_MM");
    // ---- normed bf16 operands, then the partial down projection -> cb_part2 (R tiles)
    cb_wait_front(cb_r, 2);
    cb_reserve_back(cb_hn, 2 * PER);
    for (uint32_t which = 0; which < 2; ++which) {
        const uint32_t cbx = which ? cb_m : cb_h;
        tile_regs_acquire();
        copy_from(cb_r);
        copy_tile(cb_r, which, 7);
        copy_from(cbx);
        for (uint32_t t = 0; t < PER; ++t) {
            copy_tile(cbx, t, t);
        }
        gbin_init<2>();
        for (uint32_t t = 0; t < PER; ++t) {
            gbin<2>(t, 7, t);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_hn);
        for (uint32_t t = 0; t < PER; ++t) {
            pack_tile(t, cb_hn);
        }
        tile_regs_release();
    }
    cb_push_back(cb_hn, 2 * PER);
    cb_pop_front(cb_r, 2);
    cb_wait_front(cb_hn, 2 * PER);
    cb_wait_front(cb_wd, 2 * PER * R);
    cb_reserve_back(cb_part2, R);
    tile_regs_acquire();
    mm_begin(cb_hn, cb_wd);
    for (uint32_t n = 0; n < R; ++n) {
        for (uint32_t i = 0; i < 2 * PER; ++i) {  // [h tiles | m tiles] x [Wd h rows | Wd m rows]
            matmul_tiles(cb_hn, cb_wd, i, i * R + n, n);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_part2);
    for (uint32_t n = 0; n < R; ++n) {
        pack_tile(n, cb_part2);
    }
    tile_regs_release();
    cb_push_back(cb_part2, R);
    cb_pop_front(cb_hn, 2 * PER);
    cb_pop_front(cb_wd, 2 * PER * R);

    }
    {
    DeviceZoneScopedN("G_C_ROOT2");
    // ---- root: reduce round 2 -> g (bf16, the Linear's output)
    if (root) {
        cb_wait_front(cb_gath, R * C);
        tile_regs_acquire();
        copy_from(cb_gath);
        for (uint32_t n = 0; n < R; ++n) {
            copy_tile(cb_gath, n, n);
        }
        MATH((SFPU_BINARY_INIT(unused)));
        for (uint32_t j = 1; j < C; ++j) {
            sfpu_drain();
            copy_from(cb_gath);
            for (uint32_t n = 0; n < R; ++n) {
                copy_tile(cb_gath, j * R + n, 4 + n);
            }
            for (uint32_t n = 0; n < R; ++n) {
                row_add(n, 4 + n, n);
            }
        }
        tile_regs_commit();
        cb_pop_front(cb_gath, R * C);
        cb_reserve_back(cb_res2, R);
        tile_regs_wait();
        pack_reconfig_data_format(cb_res2);
        for (uint32_t n = 0; n < R; ++n) {
            pack_tile(n, cb_res2);
        }
        tile_regs_release();
        cb_push_back(cb_res2, R);
    }

    }
    {
    DeviceZoneScopedN("G_C_FA");
    // ---- forget / gain on this core's tiles: tanh(g @ W) -> cb_fa [f tiles | a tiles] (fp32)
    cb_wait_front(cb_g, R);
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
    tanh_tile_init();
    for (uint32_t t = 0; t < PER; ++t) {
        gtanh(t);
        gtanh(4 + t);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_fa);
    for (uint32_t t = 0; t < PER; ++t) {
        pack_tile(t, cb_fa);
    }
    for (uint32_t t = 0; t < PER; ++t) {
        pack_tile(4 + t, cb_fa);
    }
    tile_regs_release();
    cb_push_back(cb_fa, 2 * PER);
    cb_pop_front(cb_g, R);
    cb_pop_front(cb_wf, R * PER);
    cb_pop_front(cb_wg, R * PER);

    }
    {
    DeviceZoneScopedN("G_C_TAIL");
    // ---- out = h - f h + a m (fp32)
    cb_wait_front(cb_fa, 2 * PER);
    for (uint32_t t = 0; t < PER; ++t) {
        cb_reserve_back(cb_out, 1);
        tile_regs_acquire();
        copy_from(cb_fa);
        copy_tile(cb_fa, t, 0);        // f
        copy_tile(cb_fa, PER + t, 1);  // a
        copy_from(cb_h);
        copy_tile(cb_h, t, 2);
        copy_from(cb_m);
        copy_tile(cb_m, t, 3);
        gbin_init<2>();
        gbin<2>(0, 2, 0);  // f h
        gbin<2>(1, 3, 1);  // a m
        gbin_init<1>();
        gbin<1>(2, 0, 2);  // h - f h
        gbin_init<0>();
        gbin<0>(2, 1, 2);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_out);
        pack_tile(2, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
    cb_pop_front(cb_fa, 2 * PER);
    cb_pop_front(cb_h, PER);
    cb_pop_front(cb_m, PER);
    }
    cb_pop_front(cb_ones, 1);
}
