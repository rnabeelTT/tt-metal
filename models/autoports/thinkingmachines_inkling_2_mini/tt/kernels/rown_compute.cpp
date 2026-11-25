// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Prefill row norm (tt/rown.py), compute: per row tile, ttnn.rms_norm's arithmetic (layernorm.cpp, RMSNORM, fp32
// DST, HiFi4) step for step, with the whole input row kept in cb_in (read once):
//   sq_i = x_i * x_i (FPU) ; s = sum_i reduce_row(sq_i) (one DST accumulator, in 4-tile blocks: the accumulator is
//   packed and restored exactly between blocks: cb_acc unpacks to DST in fp32) ; s *= 1 / W ;
//   r = rsqrt(s + eps) ; y_i = x_i * bcast_cols(r) [* bcast_rows(gamma_i)] [-> bf16 as ttnn.typecast]

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_sq = get_compile_time_arg_val(1);
    constexpr uint32_t cb_acc = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(3);
    constexpr uint32_t cb_eps = get_compile_time_arg_val(4);
    constexpr uint32_t cb_ex2 = get_compile_time_arg_val(5);
    constexpr uint32_t cb_ex2pe = get_compile_time_arg_val(6);
    constexpr uint32_t cb_fus = get_compile_time_arg_val(7);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(8);
    constexpr uint32_t cb_out = get_compile_time_arg_val(9);
    constexpr uint32_t Wt = get_compile_time_arg_val(10);
    constexpr uint32_t inv_w = get_compile_time_arg_val(11);  // bits of 1.0f / W
    constexpr uint32_t BLK = 4;
    constexpr uint32_t cb_y = get_compile_time_arg_val(12);    // ROWN_LG: the block's fp32 outputs (matmul in0)
    constexpr uint32_t cb_q = get_compile_time_arg_val(13);    // ROWN_LG: the block's Qrep tiles
    constexpr uint32_t cb_lacc = get_compile_time_arg_val(14); // ROWN_LG: the logits' running sums (exact)
    constexpr uint32_t cb_lg = get_compile_time_arg_val(15);   // ROWN_LG: the row tile's NQ logit tiles
    constexpr uint32_t NQ = get_compile_time_arg_val(16);
    const uint32_t nr = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_in, cb_in, cb_sq);
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_eps, 1);
    for (uint32_t r = 0; r < nr; ++r) {
        // ---- squares and their row sums, block by block
        for (uint32_t b = 0; b < Wt; b += BLK) {
            cb_wait_front(cb_in, b + BLK);
            reconfig_data_format(cb_in, cb_in);
            pack_reconfig_data_format(cb_sq);
            mul_init(cb_in, cb_in);
            tile_regs_acquire();
            for (uint32_t j = 0; j < BLK; ++j) {
                mul_tiles(cb_in, cb_in, b + j, b + j, j);
            }
            tile_regs_commit();
            cb_reserve_back(cb_sq, BLK);
            tile_regs_wait();
            for (uint32_t j = 0; j < BLK; ++j) {
                pack_tile(j, cb_sq);
            }
            tile_regs_release();
            cb_push_back(cb_sq, BLK);

            tile_regs_acquire();
            if (b > 0) {  // the running sum back into DST 0, exactly
                cb_wait_front(cb_acc, 1);
                reconfig_data_format_srca(cb_acc);
                copy_init(cb_acc);
                copy_tile(cb_acc, 0, 0);
                cb_pop_front(cb_acc, 1);
            }
            cb_wait_front(cb_sq, BLK);
            reconfig_data_format(cb_scaler, cb_sq);
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_scaler, cb_acc);
            for (uint32_t j = 0; j < BLK; ++j) {
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_scaler, j, 0, 0);
            }
            reduce_uninit();
            cb_pop_front(cb_sq, BLK);
            const bool last = b + BLK >= Wt;
            if (last) {
                binop_with_scalar_tile_init();
                mul_unary_tile(0, inv_w);
            }
            tile_regs_commit();
            const uint32_t ocb = last ? cb_ex2 : cb_acc;
            cb_reserve_back(ocb, 1);
            tile_regs_wait();
            pack_reconfig_data_format(ocb);
            pack_tile(0, ocb);
            tile_regs_release();
            cb_push_back(ocb, 1);
        }
        // ---- 1 / sqrt(mean + eps)
        cb_wait_front(cb_ex2, 1);
        reconfig_data_format(cb_ex2, cb_eps);
        tile_regs_acquire();
        add_init(cb_ex2, cb_eps);
        add_tiles(cb_ex2, cb_eps, 0, 0, 0);
        rsqrt_tile_init<false>();
        rsqrt_tile<false>(0);
        tile_regs_commit();
        cb_pop_front(cb_ex2, 1);
        cb_reserve_back(cb_ex2pe, 1);
        pack_reconfig_data_format(cb_ex2pe);
        tile_regs_wait();
        pack_tile(0, cb_ex2pe);
        tile_regs_release();
        cb_push_back(cb_ex2pe, 1);
        cb_wait_front(cb_ex2pe, 1);
#ifdef ROWN_GAMMA
        if (r == 0) {
            cb_wait_front(cb_gamma, Wt);  // (the reader loads it after the first row)
        }
#endif
        // ---- y = x r [gamma]
        for (uint32_t b = 0; b < Wt; b += BLK) {
            reconfig_data_format(cb_in, cb_ex2pe);
            tile_regs_acquire();
            mul_bcast_cols_init(cb_in, cb_ex2pe);
            for (uint32_t j = 0; j < BLK; ++j) {
                mul_tiles_bcast_cols(cb_in, cb_ex2pe, b + j, 0, j);
            }
#ifdef ROWN_GAMMA
            tile_regs_commit();
            cb_reserve_back(cb_fus, BLK);
            pack_reconfig_data_format(cb_fus);
            tile_regs_wait();
            for (uint32_t j = 0; j < BLK; ++j) {
                pack_tile(j, cb_fus);
            }
            tile_regs_release();
            cb_push_back(cb_fus, BLK);
            cb_wait_front(cb_fus, BLK);
            reconfig_data_format(cb_fus, cb_gamma);
            tile_regs_acquire();
            mul_bcast_rows_init(cb_fus, cb_gamma);
            for (uint32_t j = 0; j < BLK; ++j) {
                mul_tiles_bcast_rows(cb_fus, cb_gamma, j, b + j, j);
            }
            cb_pop_front(cb_fus, BLK);
#endif
#ifdef ROWN_LG
            // (no gamma, fp32 output) the outputs -> cb_out and cb_y (the logits' matmul input, as the op path's
            // matmul reads the fp32 output); then the logits sum_t y_t @ Qrep_t, the running sums kept exactly
            tile_regs_commit();
            cb_reserve_back(cb_out, BLK);
            cb_reserve_back(cb_y, BLK);
            tile_regs_wait();
            pack_reconfig_data_format(cb_out);
            for (uint32_t j = 0; j < BLK; ++j) {
                pack_tile(j, cb_out);
            }
            pack_reconfig_data_format(cb_y);
            for (uint32_t j = 0; j < BLK; ++j) {
                pack_tile(j, cb_y);
            }
            tile_regs_release();
            cb_push_back(cb_out, BLK);
            cb_push_back(cb_y, BLK);
            cb_wait_front(cb_y, BLK);
            cb_wait_front(cb_q, BLK * NQ);
            tile_regs_acquire();
            if (b > 0) {
                cb_wait_front(cb_lacc, NQ);
                reconfig_data_format_srca(cb_lacc);
                copy_init(cb_lacc);
                for (uint32_t q = 0; q < NQ; ++q) {
                    copy_tile(cb_lacc, q, q);
                }
                cb_pop_front(cb_lacc, NQ);
            }
            reconfig_data_format(cb_q, cb_y);  // matmul: in1 -> srcA, in0 -> srcB
            matmul_init(cb_y, cb_q, 0);
            for (uint32_t j = 0; j < BLK; ++j) {
                for (uint32_t q = 0; q < NQ; ++q) {
                    matmul_tiles(cb_y, cb_q, j, j * NQ + q, q);
                }
            }
            cb_pop_front(cb_q, BLK * NQ);
            cb_pop_front(cb_y, BLK);
            tile_regs_commit();
            {
                const bool lastb = b + BLK >= Wt;
                const uint32_t ocb = lastb ? cb_lg : cb_lacc;
                cb_reserve_back(ocb, NQ);
                pack_reconfig_data_format(ocb);
                tile_regs_wait();
                for (uint32_t q = 0; q < NQ; ++q) {
                    pack_tile(q, ocb);
                }
                tile_regs_release();
                cb_push_back(ocb, NQ);
            }
#else
#ifdef ROWN_BF16
            typecast_tile_init<0, 5>();
            for (uint32_t j = 0; j < BLK; ++j) {
                typecast_tile<0, 5>(j);
            }
#endif
            tile_regs_commit();
            cb_reserve_back(cb_out, BLK);
            pack_reconfig_data_format(cb_out);
            tile_regs_wait();
            for (uint32_t j = 0; j < BLK; ++j) {
                pack_tile(j, cb_out);
            }
            tile_regs_release();
            cb_push_back(cb_out, BLK);
#endif
        }
        cb_pop_front(cb_ex2pe, 1);
        cb_pop_front(cb_in, Wt);
    }
}
