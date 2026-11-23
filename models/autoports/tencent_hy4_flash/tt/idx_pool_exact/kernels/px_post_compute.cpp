// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Indexer pooling (exact), post compute (FP32 DST): keys = bf16(n * kn) (row-bcast IEEE multiply, RNE typecast:
// ttnn.multiply -> ttnn.typecast).
#include "px_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_n = 0, cb_w = 1, cb_out = 16;
    compute_kernel_hw_startup(cb_n, cb_n, cb_out);
    const uint32_t nu = get_arg_val<uint32_t>(0);
    for (uint32_t u = 0; u < nu; ++u) {
        cb_wait_front(cb_n, 1);
        cb_wait_front(cb_w, 1);
        tile_regs_acquire();
        ex::copy_to(cb_n, 0, 0);
        ex::bcast_row_to(cb_w, 0, 1);
        ex::smul(0, 1, 0);
        ex::to_bf16(0);
        tile_regs_commit();
        tile_regs_wait();
        ex::pack_to(0, cb_out);
        tile_regs_release();
        cb_pop_front(cb_n, 1);
        cb_pop_front(cb_w, 1);
    }
}
