// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Indexer pooling (exact), pool compute (FP32 DST, full sync: 8 tiles), per unit: the ttnn chain's elementwise math
// over the r = 4 slots (vg + bias, max over the slots, g - m, exp, e v, the two slot sums, div), same SFPU functions,
// same operand order (the slot sums in the order ttnn.sum(dim=1) adds them: SUM_ORDER, see ../__init__.py).
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/copy_dest_values.h"
#include "px_common.hpp"

void kernel_main() {
    constexpr uint32_t NU = get_compile_time_arg_val(0);
    constexpr uint32_t cb_vg = 0, cb_b = 1, cb_out = 16;
    compute_kernel_hw_startup(cb_vg, cb_vg, cb_out);
    const uint32_t nu = get_arg_val<uint32_t>(0);
    for (uint32_t u = 0; u < nu; ++u) {
        cb_wait_front(cb_vg, 8);
        cb_wait_front(cb_b, 8);
        tile_regs_acquire();
        // g_j (+ ape_j) -> DST j
        for (uint32_t j = 0; j < 4; ++j) {
            ex::copy_to(cb_vg, 2 * j + 1, j);
            ex::copy_to(cb_b, 2 * j + 1, 4);
            ex::sadd(j, 4, j);
        }
        // m = max over the slots -> DST 4
        copy_dest_values_init();
        copy_dest_values<DataFormat::Float32>(0, 4);
        binary_max_tile_init();
        binary_max_tile(4, 1, 4);
        binary_max_tile(4, 2, 4);
        binary_max_tile(4, 3, 4);
        // e_j = exp(g_j - m)
        for (uint32_t j = 0; j < 4; ++j) {
            sub_binary_tile_init();
            sub_binary_tile(j, 4, j);
            exp_tile_init<false>();
            exp_tile<false>(j);
        }
        // num = sum_j e_j v_j -> DST 4; den = sum_j e_j -> DST 5; ttnn.sum(dim=1) adds the slots as ((x3 + x2) + x1) + x0
        for (uint32_t jj = 0; jj < 4; ++jj) {
            const uint32_t j = 3 - jj, d = jj == 0 ? 4 : 6;
            ex::copy_to(cb_vg, 2 * j, d);
            ex::copy_to(cb_b, 2 * j, 7);
            ex::sadd(d, 7, d);
            ex::smul(j, d, d);
            if (jj > 0) {
                ex::sadd(4, 6, 4);
            }
        }
        copy_dest_values_init();
        copy_dest_values<DataFormat::Float32>(3, 5);
        ex::sadd(5, 2, 5);
        ex::sadd(5, 1, 5);
        ex::sadd(5, 0, 5);
        div_binary_tile_init();
        div_binary_tile(4, 5, 4);
        tile_regs_commit();
        tile_regs_wait();
        ex::pack_to(4, cb_out);
        tile_regs_release();
        cb_pop_front(cb_vg, 8);
        cb_pop_front(cb_b, 8);
    }
}
