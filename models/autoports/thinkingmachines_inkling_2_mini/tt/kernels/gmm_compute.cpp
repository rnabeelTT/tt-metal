// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Grouped expert matmul (tt/gmm.py), compute: per group, per row tile: [1 x Kt] @ [Kt x nb] accumulated in DST over
// the whole K (the program's fidelity, FP32 DST), packed to nb output tiles.
//
// runtime args: e0, e1 (this core's experts), G. The weight block stays at the CB front for a run of groups of one
// expert (the ids from cb_ids, sorted); padding groups (id >= E) end the scan.

#include <cstdint>

#include "api/compute/cb_api.h"
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
    constexpr uint32_t cb_ids = get_compile_time_arg_val(6);
    constexpr uint32_t E = get_compile_time_arg_val(7);
    const uint32_t e0 = get_arg_val<uint32_t>(0), e1 = get_arg_val<uint32_t>(1), G = get_arg_val<uint32_t>(2);
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
    }
    if (prev != 0xFFFFFFFF) {
        cb_pop_front(cb_w, Kt * nb);
    }
}
