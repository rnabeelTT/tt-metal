// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Merged MoE un-dispatch (tt/undisp.py), compute: per item and sub-shard s the K gathered blocks tilized and added in
// slot order 0..K-1 (as ttnn.moe_gather_add_rows), then the sub-shard sums added in order 0..S-1 (as the ttnn.add
// calls it replaces). The last add packs into the writer's buffer (S = 1: the slot sum alone, as moe_gather_add_rows).

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tilize.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

template <uint32_t Wt_c>
ALWI void add_blocks(uint32_t a, uint32_t b, uint32_t out) {
    compute_kernel_hw_startup(a, b, out);
    add_init(a, b);
    cb_wait_front(a, Wt_c);
    cb_wait_front(b, Wt_c);
    cb_reserve_back(out, Wt_c);
    for (uint32_t i0 = 0; i0 < Wt_c; i0 += 8) {
        const uint32_t n = (Wt_c - i0) < 8 ? (Wt_c - i0) : 8;
        tile_regs_acquire();
        for (uint32_t d = 0; d < n; ++d) {
            add_tiles(a, b, i0 + d, i0 + d, d);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t d = 0; d < n; ++d) {
            pack_tile(d, out);
        }
        tile_regs_release();
    }
    cb_pop_front(a, Wt_c);
    cb_pop_front(b, Wt_c);
    cb_push_back(out, Wt_c);
}

void kernel_main() {
    constexpr uint32_t Wt_c = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t S = get_compile_time_arg_val(2);
    constexpr uint32_t cb_in = get_compile_time_arg_val(3);
    constexpr uint32_t cb_til = get_compile_time_arg_val(4);
    constexpr uint32_t cb_acc_a = get_compile_time_arg_val(5);
    constexpr uint32_t cb_acc_b = get_compile_time_arg_val(6);
    constexpr uint32_t cb_tot_a = get_compile_time_arg_val(7);
    constexpr uint32_t cb_tot_b = get_compile_time_arg_val(8);
    constexpr uint32_t cb_out = get_compile_time_arg_val(9);
    const uint32_t item_count = get_arg_val<uint32_t>(0);
    static_assert(K > 1, "merged un-dispatch: several slots");

    compute_kernel_hw_startup(cb_in, cb_til, cb_acc_a);
    for (uint32_t it = 0; it < item_count; ++it) {
        uint32_t tot = cb_tot_a, tot_nxt = cb_tot_b;
        for (uint32_t s = 0; s < S; ++s) {
            compute_kernel_hw_startup(cb_in, cb_til, cb_acc_a);
            compute_kernel_lib::tilize<Wt_c, cb_in, cb_acc_a>(1);
            uint32_t cur = cb_acc_a, nxt = cb_acc_b;
            for (uint32_t j = 1; j < K; ++j) {
                // sub-shard 0's sum lands in the running total (one sub-shard: the output)
                const uint32_t dst = (j == K - 1 && s == 0) ? (S == 1 ? cb_out : tot) : nxt;
                compute_kernel_lib::tilize<Wt_c, cb_in, cb_til>(1);
                add_blocks<Wt_c>(cur, cb_til, dst);
                const uint32_t t = cur;
                cur = dst;
                nxt = t;
            }
            if (s > 0) {
                const uint32_t dst = s == S - 1 ? cb_out : tot_nxt;
                add_blocks<Wt_c>(tot, cur, dst);
                tot_nxt = tot;
                tot = dst;
            }
        }
    }
}
