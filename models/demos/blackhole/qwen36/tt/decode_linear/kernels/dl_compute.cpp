// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode linear, compute (decode_linear/ops.py DecodeLinear): per output column, sum over the kt K tiles of
// x[k] @ W[k, column] (matmul_tiles, the program's fidelity, FP32 DST), packed BF16 to cb_out.
//
// runtime args: [ncols]

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t kt = get_compile_time_arg_val(3);
    const uint32_t ncols = get_arg_val<uint32_t>(0);
    if (ncols == 0) {
        return;
    }
    compute_kernel_hw_startup(cb_x, cb_w, cb_out);
    reconfig_data_format(cb_w, cb_x);  // matmul: in1 (weights) -> srcA, in0 -> srcB
    matmul_init(cb_x, cb_w, 0);
    {
        DeviceZoneScopedN("DL_WAITX");
        cb_wait_front(cb_x, kt);
    }
    DeviceZoneScopedN("DL_MM");
    for (uint32_t c = 0; c < ncols; ++c) {
        cb_wait_front(cb_w, kt);
        tile_regs_acquire();
        for (uint32_t k = 0; k < kt; ++k) {
            matmul_tiles(cb_x, cb_w, k, k, 0);
        }
        tile_regs_commit();
        cb_pop_front(cb_w, kt);
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
    cb_pop_front(cb_x, kt);
}
