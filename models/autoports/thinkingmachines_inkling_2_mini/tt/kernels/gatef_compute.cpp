// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Prefill message-gate tail with its forget / gain matmul (tt/gatef.py), compute (fp32 DST): per output tile
//   f = g @ Wf[:, t], a = g @ Wg[:, t]   (FPU, the R K-tiles accumulated in DST as the op path's matmul)
//   out = h - tanh(f) h + tanh(a) m      (SFPU, as tt/kernels/gatet_compute.cpp)
// DST: 0 = f, 1 = a, 2 = h, 3 = m.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_g = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_hm = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t R = get_compile_time_arg_val(4);
    const uint32_t nc = get_arg_val<uint32_t>(0);
    const uint32_t nr = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_g, cb_w, cb_out);
    cb_wait_front(cb_w, 2 * R * nc);
    for (uint32_t r = 0; r < nr; ++r) {
        cb_wait_front(cb_g, R);
        for (uint32_t j = 0; j < nc; ++j) {
            cb_wait_front(cb_hm, 2);
            tile_regs_acquire();
            reconfig_data_format(cb_w, cb_g);  // matmul: in1 -> srcA, in0 -> srcB
            matmul_init(cb_g, cb_w, 0);
            for (uint32_t k = 0; k < R; ++k) {
                matmul_tiles(cb_g, cb_w, k, 2 * R * j + k, 0);
            }
            for (uint32_t k = 0; k < R; ++k) {
                matmul_tiles(cb_g, cb_w, k, 2 * R * j + R + k, 1);
            }
            reconfig_data_format_srca(cb_hm);
            copy_init(cb_hm);
            copy_tile(cb_hm, 0, 2);
            copy_tile(cb_hm, 1, 3);
            tanh_tile_init();
            tanh_tile(0);
            tanh_tile(1);
            mul_binary_tile_init();
            mul_binary_tile(0, 2, 0);  // tanh(f) h
            mul_binary_tile(1, 3, 1);  // tanh(a) m
            sub_binary_tile_init();
            sub_binary_tile(2, 0, 2);  // h - tanh(f) h
            add_binary_tile_init();
            add_binary_tile(2, 1, 2);
            tile_regs_commit();
            cb_reserve_back(cb_out, 1);
            tile_regs_wait();
            pack_tile(2, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, 1);
            cb_pop_front(cb_hm, 2);
        }
        cb_pop_front(cb_g, R);
    }
}
