// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Block attention residual mix (tt/mixw.py), compute (fp32 DST, SFPU): per query q,
//   w_i = softmax_i(l_iq)                    (elementwise over the n logit tiles: every column holds the logit)
//   out_q[t] = sum_i w_i * S_i[t]            (this core's H tiles t)
// DST (full sync, 8 fp32 tiles): softmax: 0 = max, 1..n = e_i / w_i, 7 = sum; weighted sum: 0 = acc, 1 / 2 = operands.

#include <cstdint>

#include "api/compute/binary_max_min.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_lg = get_compile_time_arg_val(0);
    constexpr uint32_t cb_s = get_compile_time_arg_val(1);
    constexpr uint32_t cb_w = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t n = get_compile_time_arg_val(4);
    constexpr uint32_t nq = get_compile_time_arg_val(5);
    const uint32_t nt = get_arg_val<uint32_t>(0);
    const uint32_t RT = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_lg, cb_out);
    for (uint32_t r = 0; r < RT; ++r) {
    cb_wait_front(cb_lg, n * nq);
    cb_wait_front(cb_s, n * nt);
    for (uint32_t q = 0; q < nq; ++q) {
        // ---- softmax weights over the n sources
        tile_regs_acquire();
        copy_init(cb_lg);
        copy_tile(cb_lg, q, 0);
        for (uint32_t i = 1; i < n; ++i) {
            copy_init(cb_lg);
            copy_tile(cb_lg, i * nq + q, 1);
            binary_max_tile_init();
            binary_max_tile(0, 1, 0);
        }
        for (uint32_t i = 0; i < n; ++i) {
            copy_init(cb_lg);
            copy_tile(cb_lg, i * nq + q, 1 + i);
            sub_binary_tile_init();
            sub_binary_tile(1 + i, 0, 1 + i);
            exp_tile_init();
            exp_tile(1 + i);
        }
        if constexpr (n == 1) {
            // a single source: w = exp(0) = 1 already
        } else {
            add_binary_tile_init();
            add_binary_tile(1, 2, 7);
            for (uint32_t i = 2; i < n; ++i) {
                add_binary_tile(7, 1 + i, 7);
            }
            recip_tile_init();
            recip_tile(7);
            mul_binary_tile_init();
            for (uint32_t i = 0; i < n; ++i) {
                mul_binary_tile(1 + i, 7, 1 + i);
            }
        }
        tile_regs_commit();
        cb_reserve_back(cb_w, n);
        tile_regs_wait();
        pack_reconfig_data_format(cb_w);
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(1 + i, cb_w);
        }
        tile_regs_release();
        cb_push_back(cb_w, n);
        cb_wait_front(cb_w, n);
        // ---- weighted sums over this core's H tiles
        for (uint32_t t = 0; t < nt; ++t) {
            tile_regs_acquire();
            copy_init(cb_s);
            copy_tile(cb_s, t * n, 0);
            copy_init(cb_w);
            copy_tile(cb_w, 0, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            for (uint32_t i = 1; i < n; ++i) {
                copy_init(cb_s);
                copy_tile(cb_s, t * n + i, 1);
                copy_init(cb_w);
                copy_tile(cb_w, i, 2);
                mul_binary_tile_init();
                mul_binary_tile(1, 2, 1);
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
            }
            tile_regs_commit();
            cb_reserve_back(cb_out, 1);
            tile_regs_wait();
            pack_reconfig_data_format(cb_out);
            pack_tile(0, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, 1);
        }
        cb_pop_front(cb_w, n);
    }
    cb_pop_front(cb_lg, n * nq);
    cb_pop_front(cb_s, n * nt);
    }
}
