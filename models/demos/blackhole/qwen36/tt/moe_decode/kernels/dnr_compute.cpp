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
#include "moe_zones.hpp"
#ifdef DN_SPLIT
#include "api/compute/eltwise_binary.h"
#endif

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
#ifdef MOE_ROWS_SKIP
    uint32_t msk[33];  // per slot, the row groups with a routed row (dnr_writer.cpp; <= 264 slots)
    for (uint32_t w = 0; w < (slots + 7) / 8; ++w) {
        uint32_t v = 0;
        UNPACK(v = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
        MATH(v = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
        PACK(v = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
        msk[w] = v;
    }
    auto live = [&](uint32_t s, uint32_t g) { return (msk[s / 8] >> (4 * (s % 8) + g)) & 1u; };
#endif
#ifdef EP_EXPERTS
    if (slots == 0) {  // none of the union on this die: the writer writes zero rows
        custom_mm_block_uninit<dense_packing>();
        return;
    }
#endif

#ifdef DN_SPLIT
    // K rows [0, DN_SPLIT) and [DN_SPLIT, kt) of every slot as two accumulation chains (DST 0.. / gc..), each packed to
    // BF16 and the two added (a two-sub-shard layout's numerics: two down programs + a BF16 add)
    constexpr uint32_t ka = DN_SPLIT, kb = kt - DN_SPLIT, gc = groups * cols;
    static_assert(ka % 2 == 0 && kb % 2 == 0 && 2 * gc <= 8, "even K segments, 8 DST tiles");
    tile_regs_acquire();
    for (uint32_t s0 = 0; s0 < slots; s0 += chunk) {
        const uint32_t cs = slots - s0 < chunk ? slots - s0 : chunk;
        cb_wait_front(cb_in0, groups * cs * kt);
#ifndef DN_SLOT
        cb_wait_front(cb_w, cs * kt * cols);
#endif
        if (s0 == 0) {
            MOE_MARK("D_C1");
        }
        for (uint32_t g = 0; g < groups; ++g) {
            for (uint32_t s = 0; s < cs; ++s) {
#ifdef DN_SLOT
                static_assert(groups == 1, "DN_SLOT: one row group");
                cb_wait_front(cb_w, kt * cols);
                const uint32_t a = (g * cs + s) * kt, w = 0;
#else
                const uint32_t a = (g * cs + s) * kt, w = s * kt * cols;
#endif
                if (s0 + s + 1 < slots) {
                    custom_mm_block<false>(cb_in0, cb_w, a, w, g * cols, ka, cols);
                    custom_mm_block<false>(cb_in0, cb_w, a + ka, w + ka * cols, gc + g * cols, kb, cols);
                } else {
                    custom_mm_block<true>(cb_in0, cb_w, a, w, g * cols, ka, cols);
                    custom_mm_block<true>(cb_in0, cb_w, a + ka, w + ka * cols, gc + g * cols, kb, cols);
                }
#ifdef DN_SLOT
                cb_pop_front(cb_w, kt * cols);
#endif
            }
        }
#ifndef DN_SLOT
        cb_pop_front(cb_w, cs * kt * cols);
#endif
        cb_pop_front(cb_in0, groups * cs * kt);
    }
    tile_regs_commit();
    MOE_MARK("D_CP");
    cb_reserve_back(DN_TMP, 2 * gc);
    tile_regs_wait();
    for (uint32_t t = 0; t < 2 * gc; ++t) {
        pack_tile(t, DN_TMP);
    }
    tile_regs_release();
    cb_push_back(DN_TMP, 2 * gc);
    custom_mm_block_uninit<dense_packing>();

    binary_op_init_common(DN_TMP, DN_TMP, cb_out);
    add_tiles_init(DN_TMP, DN_TMP);
    cb_wait_front(DN_TMP, 2 * gc);
    tile_regs_acquire();
    for (uint32_t t = 0; t < gc; ++t) {
        add_tiles(DN_TMP, DN_TMP, t, gc + t, t);
    }
    tile_regs_commit();
    cb_reserve_back(cb_out, gc);
    tile_regs_wait();
    for (uint32_t t = 0; t < gc; ++t) {
        pack_tile(t, cb_out);
    }
    tile_regs_release();
    cb_push_back(cb_out, gc);
    cb_pop_front(DN_TMP, 2 * gc);
    return;
#endif
    tile_regs_acquire();
    for (uint32_t s0 = 0; s0 < slots; s0 += chunk) {
        const uint32_t cs = slots - s0 < chunk ? slots - s0 : chunk;
        cb_wait_front(cb_in0, groups * cs * kt);
#ifdef DN_SLOT
        static_assert(groups == 1, "DN_SLOT: one row group");
        for (uint32_t s = 0; s < cs; ++s) {
            cb_wait_front(cb_w, kt * cols);
            if (s0 + s == 0) {
                MOE_MARK("D_C1");
            }
            if (s0 + s + 1 < slots) {
                custom_mm_block<false>(cb_in0, cb_w, s * kt, 0, 0, kt, cols);
            } else {
                custom_mm_block<true>(cb_in0, cb_w, s * kt, 0, 0, kt, cols);
            }
            cb_pop_front(cb_w, kt * cols);
        }
        cb_pop_front(cb_in0, groups * cs * kt);
        continue;
#endif
        cb_wait_front(cb_w, cs * kt * cols);
        if (s0 == 0) {
            MOE_MARK("D_C1");
        }
#ifdef MOE_ROWS_SKIP
        // each live (slot, group) as its own K run (the skipped ones were exact zeros); the group's last live slot
        // finalizes (every group has one: each of its rows routed to k experts of the union)
        for (uint32_t g = 0; g < groups; ++g) {
            for (uint32_t s = 0; s < cs; ++s) {
                if (!live(s0 + s, g)) {
                    continue;
                }
                bool last = true;
                for (uint32_t t = s0 + s + 1; t < slots; ++t) {
                    if (live(t, g)) {
                        last = false;
                        break;
                    }
                }
                if (last) {
                    custom_mm_block<true>(cb_in0, cb_w, (g * cs + s) * kt, s * kt * cols, g * cols, kt, cols);
                } else {
                    custom_mm_block<false>(cb_in0, cb_w, (g * cs + s) * kt, s * kt * cols, g * cols, kt, cols);
                }
            }
        }
#else
        for (uint32_t g = 0; g < groups; ++g) {
            if (s0 + cs < slots) {
                custom_mm_block<false>(cb_in0, cb_w, g * cs * kt, 0, g * cols, cs * kt, cols);
            } else {
                custom_mm_block<true>(cb_in0, cb_w, g * cs * kt, 0, g * cols, cs * kt, cols);
            }
        }
#endif
        cb_pop_front(cb_w, cs * kt * cols);
        cb_pop_front(cb_in0, groups * cs * kt);
    }
    tile_regs_commit();
    MOE_MARK("D_CP");

    cb_reserve_back(cb_out, groups * cols);
    tile_regs_wait();
    for (uint32_t t = 0; t < groups * cols; ++t) {
        pack_tile(t, cb_out);
    }
    tile_regs_release();
    cb_push_back(cb_out, groups * cols);
    custom_mm_block_uninit<dense_packing>();
}
