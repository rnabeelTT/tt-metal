// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Per unit: DST 0 = sum_k x_k @ fn_k (projection partial), DST 1 = sum_k x_k @ x_k^T (Gram partial; its diagonal is the
// row sum of squares over the chunk). fp32 DST, HiFi4.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t Kc = get_compile_time_arg_val(0);
    const uint32_t num_units = get_arg_val<uint32_t>(0);
    constexpr uint32_t CB_X = 0, CB_FN = 1, CB_OUT = 16;
    if (num_units == 0) {
        return;
    }
    compute_kernel_hw_startup(CB_X, CB_FN, CB_OUT);
    CircularBuffer cx(CB_X), cfn(CB_FN), co(CB_OUT);
    for (uint32_t w = 0; w < num_units; ++w) {
        cx.wait_front(Kc);
        cfn.wait_front(Kc);
        co.reserve_back(2);
        tile_regs_acquire();
        matmul_init(CB_X, CB_FN, 0);
        for (uint32_t k = 0; k < Kc; ++k) {
            matmul_tiles(CB_X, CB_FN, k, k, 0);
        }
        matmul_init(CB_X, CB_X, 1);
        for (uint32_t k = 0; k < Kc; ++k) {
            matmul_tiles(CB_X, CB_X, k, k, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_OUT);
        pack_tile(1, CB_OUT);
        tile_regs_release();
        co.push_back(2);
        cx.pop_front(Kc);
        cfn.pop_front(Kc);
    }
}
