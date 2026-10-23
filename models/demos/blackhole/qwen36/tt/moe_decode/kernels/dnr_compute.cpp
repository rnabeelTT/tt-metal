// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Multi-row down projection + weighted expert sum (TRISC) of the fused MoE decode (moe_decode/union.py
// ExpertDownRows): per row group g, out_g[:, cols] = act_g ([trows, (U + 1) K], slot-major) . Wstack ([(U + 1) K, cols]),
// custom_mm per chunk of `chunk` slots accumulating in DST tiles g * cols .. (finalize on the last chunk; groups * cols
// <= 4 FP32 DST tiles, 8 in full-sync mode (DST_FULL): this kernel packs once, at the end), packed group-major as [trows, 32] tiles. The slot count comes from BRISC (mailboxes).

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
    constexpr uint32_t groups = get_compile_time_arg_val(6);
#ifdef DST_FULL
    static_assert(groups * cols <= 8, "FP32 DST (full sync) holds 8 output tiles");
#else
    static_assert(groups * cols <= 4, "FP32 DST holds 4 output tiles");
#endif

    constexpr bool transpose = false;
    constexpr bool split_acc = true;
    constexpr bool dense_packing = false;
    custom_mm_block_init<transpose, split_acc, dense_packing>(cb_in0, cb_w, cb_out, cols);

    uint32_t slots = 0;
    UNPACK(slots = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    MATH(slots = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    PACK(slots = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
#ifdef EP_EXPERTS
    if (slots == 0) {  // none of the union on this die: the writer writes zero rows
        custom_mm_block_uninit<dense_packing>();
        return;
    }
#endif

    tile_regs_acquire();
    for (uint32_t s0 = 0; s0 < slots; s0 += chunk) {
        const uint32_t cs = slots - s0 < chunk ? slots - s0 : chunk;
        cb_wait_front(cb_in0, groups * cs * kt);
        cb_wait_front(cb_w, cs * kt * cols);
        for (uint32_t g = 0; g < groups; ++g) {
            if (s0 + cs < slots) {
                custom_mm_block<false>(cb_in0, cb_w, g * cs * kt, 0, g * cols, cs * kt, cols);
            } else {
                custom_mm_block<true>(cb_in0, cb_w, g * cs * kt, 0, g * cols, cs * kt, cols);
            }
        }
        cb_pop_front(cb_w, cs * kt * cols);
        cb_pop_front(cb_in0, groups * cs * kt);
    }
    tile_regs_commit();

    cb_reserve_back(cb_out, groups * cols);
    tile_regs_wait();
    for (uint32_t t = 0; t < groups * cols; ++t) {
        pack_tile(t, cb_out);
    }
    tile_regs_release();
    cb_push_back(cb_out, groups * cols);
    custom_mm_block_uninit<dense_packing>();
}
