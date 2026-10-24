// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Attention decode post, compute: attn_out * gate per tile (bf16 in, FP32 DST, bf16 out).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t cb_a = 0, cb_g = 1, cb_y = 2;
    const uint32_t count = get_arg_val<uint32_t>(0);
    if (count == 0) {
        return;
    }
    compute_kernel_hw_startup(cb_a, cb_a, cb_y);
    CircularBuffer a(cb_a), g(cb_g), y(cb_y);
    copy_init(cb_a);
    mul_binary_tile_init();
    for (uint32_t i = 0; i < count; i++) {
        a.wait_front(Ht);
        g.wait_front(Ht);
        y.reserve_back(Ht);
        for (uint32_t j = 0; j < Ht; j++) {
            tile_regs_acquire();
            copy_tile(cb_a, j, 0);
            copy_tile(cb_g, j, 1);
            mul_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_y);
            tile_regs_release();
        }
        y.push_back(Ht);
        a.pop_front(Ht);
        g.pop_front(Ht);
    }
}
