// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Multi-row down projection + weighted expert sum (TRISC) of the fused MoE decode (moe_decode/union.py
// ExpertDownRows): out[:, cols] = act ([M, (U + 1) K], slot-major) . Wstack ([(U + 1) K, cols]), custom_mm per chunk
// of `chunk` slots accumulating in DST (finalize on the last chunk), packed as `cols` [M, 32] tiles. The slot count
// comes from BRISC through the mailboxes.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/pack.h"

void kernel_main() {
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t kt = get_compile_time_arg_val(3);  // K tiles per slot
    constexpr uint32_t cols = get_compile_time_arg_val(4);
    constexpr uint32_t chunk = get_compile_time_arg_val(5);

    constexpr bool transpose = false;
    constexpr bool split_acc = true;
    constexpr bool dense_packing = false;
    custom_mm_block_init<transpose, split_acc, dense_packing>(cb_in0, cb_w, cb_out, cols);

    uint32_t slots = 0;
    UNPACK(slots = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    MATH(slots = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    PACK(slots = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));

    cb_wait_front(cb_in0, slots * kt);
    tile_regs_acquire();
    for (uint32_t s0 = 0; s0 < slots; s0 += chunk) {
        const uint32_t cs = slots - s0 < chunk ? slots - s0 : chunk;
        cb_wait_front(cb_w, cs * kt * cols);
        if (s0 + cs < slots) {
            custom_mm_block<false>(cb_in0, cb_w, s0 * kt, 0, 0, cs * kt, cols);
        } else {
            custom_mm_block<true>(cb_in0, cb_w, s0 * kt, 0, 0, cs * kt, cols);
        }
        cb_pop_front(cb_w, cs * kt * cols);
    }
    tile_regs_commit();
    cb_pop_front(cb_in0, slots * kt);

    cb_reserve_back(cb_out, cols);
    tile_regs_wait();
    for (uint32_t c = 0; c < cols; ++c) {
        pack_tile(c, cb_out);
    }
    tile_regs_release();
    cb_push_back(cb_out, cols);
    custom_mm_block_uninit<dense_packing>();
}
