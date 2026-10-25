// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode layer boundary, compute (TRISC) (fabric_boundary/ops.py: FabricBoundary). From the gpt-oss fused decode
// (models/demos/gpt_oss/tt/fused_decode/kernels/boundary_compute.cpp); this image's calculate_rsqrt takes one more template flag.
//
// The decode residual stream is a flat BF16 vector stored in `tiles` 32x32 tile pages (hidden value h at byte 2 h,
// zero padded). Every op here is elementwise or a whole-vector sum, so the tile structure is only a container:
//   1. residual' = (sum of the `slots` partial sums in slot order) + residual: the same order on every device of
//      the TP ring (bit-identical replicated residual), the partials first so the BF16 rounding of the running sum
//      happens at their magnitude; packed into cb_res_out (backed by the output residual);
//   2. ss = sum(residual'^2) (FP32, squares of the BF16-rounded residual, as the HF RMSNorm sees it);
//   3. x = (residual' * gamma) * rsqrt(ss / hidden + eps), packed into cb_x (backed by the normed output).
// All tile loads go through the FPU (unpack A/B); the SFPU only works on DST. The FPU phases reading the FP32
// intermediates (the scalar reduce, the scalar broadcast) start with a full compute_kernel_hw_startup (as groupnorm
// does mid kernel): with only the op inits the reduce sums part of the tile. BND_* zones: device-profiler breakdown
// (compiled out without the profiler).

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/pack.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "tools/profiler/kernel_profiler.hpp"

// SFPU ops on the scalar statistic in element (0, 0) of DST: first rows of faces 0 / 1 only (as the gate|up stream's
// row ops), not the whole tile.
constexpr uint32_t kScalarIters = 1;
template <int OP>
ALWI void scalar_op(uint32_t idst, uint32_t s) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binop_with_scalar,
        (APPROX, OP, kScalarIters, DST_ACCUM_MODE),
        idst,
        VectorMode::R,
        s));
}
ALWI void rsqrt_scalar(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_rsqrt,
        (APPROX, kScalarIters, DST_ACCUM_MODE, false, false),
        idst,
        VectorMode::R));
}

void kernel_main() {
    constexpr uint32_t cb_res = get_compile_time_arg_val(0);
    constexpr uint32_t cb_recv = get_compile_time_arg_val(1);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(3);
    constexpr uint32_t cb_res_out = get_compile_time_arg_val(4);
    constexpr uint32_t cb_sq = get_compile_time_arg_val(5);
    constexpr uint32_t cb_rs = get_compile_time_arg_val(6);
    constexpr uint32_t cb_x = get_compile_time_arg_val(7);
    constexpr uint32_t tiles = get_compile_time_arg_val(8);
    constexpr uint32_t slots = get_compile_time_arg_val(9);
    constexpr uint32_t inv_n = get_compile_time_arg_val(10);  // float bits of 1 / hidden
    constexpr uint32_t eps = get_compile_time_arg_val(11);    // float bits
    constexpr uint32_t do_norm = get_compile_time_arg_val(12);
    constexpr uint32_t cb_y = get_compile_time_arg_val(13);
    constexpr uint32_t rows = get_compile_time_arg_val(14);  // decode rows: row r's vector at pages r * tiles ..
    constexpr uint32_t all = rows * tiles;
    static_assert(tiles <= 4 && 4 % tiles == 0, "rows of whole tiles in the 4 FP32 DST tiles");

    // 1. residual' = ((slot 0 + slot 1) + ...) + residual: the partial sums first (rounded at their own magnitude
    //    when DST feeds SrcA), the residual last (FPU, FP32 DST); 4 tiles per DST acquire.
    compute_kernel_hw_startup(slots > 0 ? cb_recv : cb_res, cb_res, cb_res_out);
    cb_wait_front(cb_res, all);
    if constexpr (slots > 0) {
        cb_wait_front(cb_recv, slots * all);
    }
    cb_reserve_back(cb_res_out, all);
    {
        DeviceZoneScopedN("BND_C1_SUM");
        for (uint32_t t0 = 0; t0 < all; t0 += 4) {
            const uint32_t n = all - t0 < 4 ? all - t0 : 4;
            tile_regs_acquire();
            for (uint32_t j = 0; j < n; ++j) {
                const uint32_t t = t0 + j;
                if constexpr (slots > 1) {
                    add_init(cb_recv, cb_recv);
                    add_tiles(cb_recv, cb_recv, t, all + t, j);
                    add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_recv);
                    for (uint32_t s = 2; s < slots; ++s) {
                        add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_recv, s * all + t, j);
                    }
                    add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_res);
                    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_res, t, j);
                } else if constexpr (slots == 1) {
                    add_init(cb_recv, cb_res);
                    add_tiles(cb_recv, cb_res, t, t, j);
                } else {
                    copy_init(cb_res);
                    copy_tile(cb_res, t, j);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < n; ++j) {
                pack_tile(j, cb_res_out);
            }
            tile_regs_release();
        }
    }
    cb_push_back(cb_res_out, all);
    cb_pop_front(cb_res, all);
    if constexpr (slots > 0) {
        cb_pop_front(cb_recv, slots * all);
    }
    if constexpr (!do_norm) {
        return;
    }

    // 2. per row: y = residual' * gamma (FP32, packed to cb_y) and the per-position sum of residual'^2 over the row's
    //    tiles (packed to cb_sq, one tile per row); RG = 4 / tiles rows per DST acquire.
    constexpr uint32_t RG = 4 / tiles;
    cb_wait_front(cb_res_out, all);
    cb_wait_front(cb_gamma, tiles);
    cb_wait_front(cb_scaler, 1);
    compute_kernel_hw_startup(cb_res_out, cb_gamma, cb_y);
    pack_reconfig_data_format(cb_y);
    {
        DeviceZoneScopedN("BND_C2_SQ");
        for (uint32_t r0 = 0; r0 < rows; r0 += RG) {
            const uint32_t nr = rows - r0 < RG ? rows - r0 : RG;
            tile_regs_acquire();
            mul_init(cb_res_out, cb_gamma);
            for (uint32_t r = 0; r < nr; ++r) {
                for (uint32_t t = 0; t < tiles; ++t) {
                    mul_tiles(cb_res_out, cb_gamma, (r0 + r) * tiles + t, t, r * tiles + t);
                }
            }
            tile_regs_commit();
            cb_reserve_back(cb_y, nr * tiles);
            tile_regs_wait();
            for (uint32_t i = 0; i < nr * tiles; ++i) {
                pack_tile(i, cb_y);
            }
            tile_regs_release();
            cb_push_back(cb_y, nr * tiles);

            tile_regs_acquire();
            mul_init(cb_res_out, cb_res_out);
            for (uint32_t r = 0; r < nr; ++r) {
                for (uint32_t t = 0; t < tiles; ++t) {
                    const uint32_t i = (r0 + r) * tiles + t;
                    mul_tiles(cb_res_out, cb_res_out, i, i, r * tiles + t);
                }
            }
            add_binary_tile_init();
            for (uint32_t r = 0; r < nr; ++r) {
                for (uint32_t t = 1; t < tiles; ++t) {
                    add_binary_tile(r * tiles, r * tiles + t, r * tiles);
                }
            }
            tile_regs_commit();
            cb_reserve_back(cb_sq, nr);
            tile_regs_wait();
            for (uint32_t r = 0; r < nr; ++r) {
                pack_tile(r * tiles, cb_sq);
            }
            tile_regs_release();
            cb_push_back(cb_sq, nr);
        }
    }

    // 3. rs = rsqrt(sum / hidden + eps) per row: FPU scalar reduce, SFPU scalar ops; 4 rows per acquire.
    cb_wait_front(cb_sq, rows);
    compute_kernel_hw_startup(cb_sq, cb_scaler, cb_rs);
    {
        DeviceZoneScopedN("BND_C3_RS");
        for (uint32_t r0 = 0; r0 < rows; r0 += 4) {
            const uint32_t nr = rows - r0 < 4 ? rows - r0 : 4;
            tile_regs_acquire();
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_sq, cb_scaler, cb_rs);
            for (uint32_t r = 0; r < nr; ++r) {
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_sq, cb_scaler, r0 + r, 0, r);
            }
            reduce_uninit(cb_sq);
            binop_with_scalar_tile_init();
            for (uint32_t r = 0; r < nr; ++r) {
                scalar_op<MUL_UNARY>(r, inv_n);
                scalar_op<ADD_UNARY>(r, eps);
            }
            rsqrt_tile_init();
            for (uint32_t r = 0; r < nr; ++r) {
                rsqrt_scalar(r);
            }
            tile_regs_commit();
            cb_reserve_back(cb_rs, nr);
            tile_regs_wait();
            for (uint32_t r = 0; r < nr; ++r) {
                pack_tile(r, cb_rs);
            }
            tile_regs_release();
            cb_push_back(cb_rs, nr);
        }
    }
    cb_pop_front(cb_sq, rows);

    // 4. x = y * rs per row (FPU scalar broadcast); RG rows per acquire.
    cb_wait_front(cb_rs, rows);
    cb_wait_front(cb_y, all);
    cb_reserve_back(cb_x, all);
    compute_kernel_hw_startup(cb_y, cb_rs, cb_x);
    pack_reconfig_data_format(cb_x);
    {
        DeviceZoneScopedN("BND_C4_X");
        mul_bcast_scalar_init(cb_y, cb_rs);
        for (uint32_t r0 = 0; r0 < rows; r0 += RG) {
            const uint32_t nr = rows - r0 < RG ? rows - r0 : RG;
            tile_regs_acquire();
            for (uint32_t r = 0; r < nr; ++r) {
                for (uint32_t t = 0; t < tiles; ++t) {
                    mul_tiles_bcast_scalar(cb_y, cb_rs, (r0 + r) * tiles + t, r0 + r, r * tiles + t);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < nr * tiles; ++i) {
                pack_tile(i, cb_x);
            }
            tile_regs_release();
        }
    }
    cb_push_back(cb_x, all);
    cb_pop_front(cb_y, all);
    cb_pop_front(cb_rs, rows);
    cb_pop_front(cb_gamma, tiles);
    cb_pop_front(cb_scaler, 1);
}
