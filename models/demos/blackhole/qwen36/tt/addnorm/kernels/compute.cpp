// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Decode add + RMSNorm, compute (one core per residual shard of T tiles, rows = users):
//   h = residual (+ all-reduce result when ADD) -> cb_h (bf16, the new residual);
//   P = sum_t (h_t * h_t) @ ones (FP32: every column = this shard's per-row sum of squares) -> cb_p (the writer sends
//     it to every core);
//   r = rsqrt(sum of the NC shards' P / D + eps) (the writer pushes the NC received tiles to cb_slots);
//   out = h * r (column 0 broadcast) * gamma (DST reuse, every row = gamma) -> cb_out (bf16).

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr uint32_t ADD = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(3);
    constexpr uint32_t inv_d_bits = get_compile_time_arg_val(4);
    constexpr uint32_t cb_res = 0, cb_red = 1, cb_w = 2, cb_ones = 3, cb_h = 4, cb_sq = 5, cb_p = 6, cb_slots = 7,
                       cb_r = 8, cb_out = 9;
    constexpr uint32_t cb_x = ADD ? cb_h : cb_res;  // h

    compute_kernel_hw_startup(cb_res, cb_res, cb_out);
    CircularBuffer res(cb_res), h(cb_h), sq(cb_sq), p(cb_p), slots(cb_slots), r(cb_r), out(cb_out), w(cb_w),
        ones(cb_ones);
    res.wait_front(T);
    if constexpr (ADD) {
        CircularBuffer red(cb_red);
        red.wait_front(T);
        h.reserve_back(T);
        pack_reconfig_data_format(cb_h);
        reconfig_data_format(cb_res, cb_red);
        add_tiles_init(cb_res, cb_red);
        for (uint32_t t0 = 0; t0 < T; t0 += 4) {
            tile_regs_acquire();
            for (uint32_t t = t0; t < t0 + 4 && t < T; t++) {
                add_tiles(cb_res, cb_red, t, t, t - t0);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t t = t0; t < t0 + 4 && t < T; t++) {
                pack_tile(t - t0, cb_h);
            }
            tile_regs_release();
        }
        h.push_back(T);
        red.pop_front(T);
        h.wait_front(T);
    }
    // squares (FP32)
    sq.reserve_back(T);
    pack_reconfig_data_format(cb_sq);
    reconfig_data_format(cb_x, cb_x);
    mul_init(cb_x, cb_x);
    for (uint32_t t0 = 0; t0 < T; t0 += 4) {
        tile_regs_acquire();
        for (uint32_t t = t0; t < t0 + 4 && t < T; t++) {
            mul_tiles(cb_x, cb_x, t, t, t - t0);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t t = t0; t < t0 + 4 && t < T; t++) {
            pack_tile(t - t0, cb_sq);
        }
        tile_regs_release();
    }
    sq.push_back(T);
    // this shard's per-row sum of squares, every column
    ones.wait_front(1);
    sq.wait_front(T);
    p.reserve_back(1);
    pack_reconfig_data_format(cb_p);
    tile_regs_acquire();
    reconfig_data_format(cb_ones, cb_sq);  // matmul: in1 -> srcA, in0 -> srcB
    matmul_init(cb_sq, cb_ones, 0);
    for (uint32_t t = 0; t < T; t++) {
        matmul_tiles(cb_sq, cb_ones, t, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_p);
    tile_regs_release();
    p.push_back(1);
    sq.pop_front(T);
    // r from the NC shards' sums
    slots.wait_front(NC);
    r.reserve_back(1);
    pack_reconfig_data_format(cb_r);
    tile_regs_acquire();
    reconfig_data_format(cb_slots, cb_slots);
    add_tiles_init(cb_slots, cb_slots);
    add_tiles(cb_slots, cb_slots, 0, 1, 0);
    add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_slots);
    for (uint32_t s = 2; s < NC; s++) {
        add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_slots, s, 0);
    }
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_d_bits);
    add_unary_tile(0, eps_bits);
    rsqrt_tile_init<false>();
    rsqrt_tile<false, false>(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_r);
    tile_regs_release();
    r.push_back(1);
    slots.pop_front(NC);
    // out = h r gamma
    r.wait_front(1);
    w.wait_front(T);
    out.reserve_back(T);
    pack_reconfig_data_format(cb_out);
    for (uint32_t t = 0; t < T; t++) {
        tile_regs_acquire();
        reconfig_data_format(cb_x, cb_r);
        mul_bcast_cols_init(cb_x, cb_r);
        mul_tiles_bcast_cols(cb_x, cb_r, t, 0, 0);
        reconfig_data_format_srcb(cb_w);
        mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_w);
        mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_w, t, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
    }
    out.push_back(T);
    r.pop_front(1);
    w.pop_front(T);
    ones.pop_front(1);
    res.pop_front(T);
    if constexpr (ADD) {
        h.pop_front(T);
    }
}
