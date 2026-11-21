// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fused RoPE, compute: per x tile, in FP32 DST (the same arithmetic as tt/rope.DeviceRope's 6 ttnn ops):
//   rh = x @ P (FPU, the +-1 rotate-half permutation: exact), y = bf16_rne(x * cos + rh * sin) (SFPU FP32 mul / add,
//   the RNE narrowing of ttnn.typecast).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_x = 0, cb_p = 1, cb_c = 2, cb_s = 3, cb_y = 16;
    const uint32_t n = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_x, cb_p, cb_y);
    cb_wait_front(cb_p, 1);
    for (uint32_t t = 0; t < n; ++t) {
        cb_wait_front(cb_x, 1);
        cb_wait_front(cb_c, 1);
        cb_wait_front(cb_s, 1);
        cb_reserve_back(cb_y, 1);
        pack_reconfig_data_format(cb_y);
        tile_regs_acquire();
        reconfig_data_format(cb_p, cb_x);
        matmul_init(cb_x, cb_p, 0);
        matmul_tiles(cb_x, cb_p, 0, 0, 0);  // DST0 = rotate_half(x)
        reconfig_data_format_srca(cb_x);
        copy_init(cb_x);
        copy_tile(cb_x, 0, 1);  // DST1 = x
        reconfig_data_format_srca(cb_c);
        copy_init(cb_c);
        copy_tile(cb_c, 0, 2);  // DST2 = cos (FP32 unpack to DST)
        mul_binary_tile_init();
        mul_binary_tile(1, 2, 1);
        reconfig_data_format_srca(cb_s);
        copy_init(cb_s);
        copy_tile(cb_s, 0, 2);  // DST2 = sin
        mul_binary_tile_init();
        mul_binary_tile(0, 2, 0);
        add_binary_tile_init();
        add_binary_tile(1, 0, 1);
        typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>();
        typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(1, cb_y);
        tile_regs_release();
        cb_push_back(cb_y, 1);
        cb_pop_front(cb_x, 1);
        cb_pop_front(cb_c, 1);
        cb_pop_front(cb_s, 1);
    }
    cb_pop_front(cb_p, 1);
}
