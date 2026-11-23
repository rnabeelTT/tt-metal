// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Grouped expert matmul (tt/gmm.py), compute: per group, per row tile: [1 x Kt] @ [Kt x nb] accumulated in DST over
// the whole K (the program's fidelity, FP32 DST), packed to nb output tiles.
//
// runtime args: ng (groups of this core)

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t Kt = get_compile_time_arg_val(3);
    constexpr uint32_t nb = get_compile_time_arg_val(4);
    constexpr uint32_t mt = get_compile_time_arg_val(5);
    const uint32_t ng = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_w, cb_out);  // (matmul: in1 -> srcA, the stock bmm's order)
    matmul_block_init(cb_a, cb_w, false, nb, 1, Kt);
#if GMM_RM
    pack_untilize_dest_init<nb>(cb_out);  // (row-major output: 32 rows x nb tiles per block)
#endif
    for (uint32_t g = 0; g < ng; ++g) {
        cb_wait_front(cb_w, Kt * nb);
        for (uint32_t mi = 0; mi < mt; ++mi) {
            cb_wait_front(cb_a, Kt);
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                matmul_block(cb_a, cb_w, kt, kt * nb, 0, false, nb, 1, Kt);
            }
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
        cb_pop_front(cb_w, Kt * nb);
    }
}
