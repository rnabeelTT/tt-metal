// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fused packed decode all-reduce (tt/ccl.py _Packed.fused): the sum of the N slots (KJ tiles each) into the output
// shard. The same instruction sequence as all_reduce_async's reduction.cpp (DST accumulation of slot pairs,
// (s0 + s1) + (s2 + s3)): bit-identical sums.

#include <cstdint>

#include "api/compute/eltwise_binary.h"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t N = get_compile_time_arg_val(2);  // (even)
    constexpr uint32_t KJ = get_compile_time_arg_val(3);  // <= 8
    cb_wait_front(cb_in, N * KJ);
    cb_reserve_back(cb_out, KJ);
    compute_kernel_hw_startup(cb_in, cb_in, cb_out);
    add_init(cb_in, cb_in, true);
    tile_regs_acquire();
    for (uint32_t b = 0; b < N; b += 2) {
        for (uint32_t i = 0; i < KJ; ++i) {
            add_tiles(cb_in, cb_in, b * KJ + i, (b + 1) * KJ + i, i);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < KJ; ++i) {
        pack_tile(i, cb_out, i);
    }
    tile_regs_release();
    cb_push_back(cb_out, KJ);
}
