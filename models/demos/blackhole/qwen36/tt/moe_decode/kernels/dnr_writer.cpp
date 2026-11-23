// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Multi-row down-projection activation gather + output scatter (BRISC, NOC1) of the fused MoE decode
// (moe_decode/union.py ExpertDownRows): M rows as `groups` row groups of [trows, 32] tiles.
//
// 1. Builds the union from the route table (the slot count, to the three compute threads through the mailboxes).
// 2. in0, chunk by chunk of `chunk` union slots (the reader streams the weights in the same chunks): the activation
//    buffer's tiles (page (s * kt + k) * groups + g) group-major, then slot, then K row: cb_in0 tile
//    g * cs * kt + (s - s0) * kt + k of the chunk.
// 3. Writes each [trows, 32] output tile (group g, column n = bank + banks * (c0 + c)) into rows g * trows .. of tile
//    n of the [32, H] BF16 TILE output (row r at byte (r / 16) * 1024 + (r % 16) * 32; + 512 for face 1).
//
// NO_SHARED: routed experts only (union slots 1 .. U).
//
// EP_EXPERTS: only this die's window of the union (local ids; 0 slots -> zero output rows).
//
// GUDN (union.py ExpertGateUpDownRows): the gate|up cores run in the same program; a chunk's activations are read once
// the gate|up cores' progress words cover its slots.
//
// runtime args: [act_addr, out_addr, table_addr, bank, c0, (EP_EXPERTS) ep_offset_addr, (GUDN) gate|up core count,
//                (x | y << 8 | first slot << 16) per gate|up core]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "router_topk.hpp"

void kernel_main() {
    const uint32_t act_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t table_addr = get_arg_val<uint32_t>(2);
    const uint32_t bank = get_arg_val<uint32_t>(3);
    const uint32_t c0 = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t cb_tab = get_compile_time_arg_val(2);
    constexpr uint32_t kt = get_compile_time_arg_val(3);
    constexpr uint32_t cols = get_compile_time_arg_val(4);
    constexpr uint32_t banks = get_compile_time_arg_val(5);
    constexpr uint32_t topk = get_compile_time_arg_val(6);
    constexpr uint32_t rows = get_compile_time_arg_val(7);
    constexpr uint32_t trows = get_compile_time_arg_val(8);
    constexpr uint32_t groups = get_compile_time_arg_val(9);
    constexpr uint32_t chunk = get_compile_time_arg_val(10);
    constexpr uint32_t act_page = get_compile_time_arg_val(11);  // activation buffer page (>= one tile)
    constexpr auto act_args = TensorAccessorArgs<12>();
    constexpr auto out_args = TensorAccessorArgs<act_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    constexpr uint32_t tb = trows * 64;
    constexpr uint32_t half = trows * 32;

    const auto s_act = TensorAccessor(act_args, act_addr, act_page);
    const auto s_out = TensorAccessor(out_args, out_addr, 2048);
    const auto s_table = TensorAccessor(table_args, table_addr, moe_topk::REC_BYTES);

    cb_reserve_back(cb_tab, 1);
    const uint32_t tab = get_write_ptr(cb_tab);
    for (uint32_t r = 0; r < rows; ++r) {
        noc_async_read(s_table.get_noc_addr(r), tab + moe_topk::REC_BYTES * r, moe_topk::REC_BYTES);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    uint16_t uni[rows * topk];
#ifdef NO_SHARED
    constexpr uint32_t S0 = 1;  // no shared expert: slots 1 .. (routed only)
#else
    constexpr uint32_t S0 = 0;
#endif
#ifdef EP_EXPERTS
    const uint32_t slots = 1 - S0 + moe_topk::window_union(
        uni, moe_topk::route_union<topk>(reinterpret_cast<const uint16_t*>(tab), rows, uni),
        moe_topk::read_ep_offset(get_arg_val<uint32_t>(5), tab + 64 * rows), EP_EXPERTS);
#else
    const uint32_t slots = 1 - S0 + moe_topk::route_union<topk>(reinterpret_cast<const uint16_t*>(tab), rows, uni);
#endif
    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, slots);
    ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, slots);
    ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, slots);
#ifdef EP_EXPERTS
    if (slots == 0) {  // none of the union on this die: zero partial rows
        for (uint32_t g = 0; g < groups; ++g) {
            const uint32_t r0 = g * trows;
            const uint32_t roff = (r0 / 16) * 1024 + (r0 % 16) * 32;
            for (uint32_t c = 0; c < cols; ++c) {
                const uint32_t n = bank + banks * (c0 + c);
                noc_async_write(MEM_ZEROS_BASE, s_out.get_noc_addr(n, roff), half);
                noc_async_write(MEM_ZEROS_BASE, s_out.get_noc_addr(n, roff + 512), half);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_tab, 1);
        return;
    }
#endif

#ifdef GUDN
    // gate|up + down in one program (union.py ExpertGateUpDownRows): before a chunk's activation reads, the progress
    // words of the gate|up cores (runtime args: count, then NoC x | y << 8 | first slot << 16 each) must cover every
    // union slot < the chunk's end (a core owns slots first, first + gstep, ...; it counts drained slots in order)
    constexpr uint32_t gudn_ct = table_args.next_compile_time_args_offset();
    constexpr uint32_t cb_prog = get_compile_time_arg_val(gudn_ct);
    constexpr uint32_t gstep = get_compile_time_arg_val(gudn_ct + 1);
#ifdef EP_EXPERTS
    constexpr uint32_t gudn_rt = 6;
#else
    constexpr uint32_t gudn_rt = 5;
#endif
    constexpr uint32_t max_gu = 64;
    const uint32_t n_gu = get_arg_val<uint32_t>(gudn_rt);
    const uint32_t prog_addr = get_write_ptr(cb_prog);  // (the gate|up cores' word: same address there)
    uint32_t known[max_gu];  // progress words read so far (monotonic)
    for (uint32_t c = 0; c < n_gu; ++c) {
        known[c] = 0;
    }
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2)), n_gu);
    auto wait_slots = [&](uint32_t end, uint32_t scratch) {  // union slots < end written by every gate|up core
        while (true) {
            uint32_t pending = 0;
            for (uint32_t c = 0; c < n_gu; ++c) {
                const uint32_t w = get_arg_val<uint32_t>(gudn_rt + 1 + c);
                const uint32_t first = w >> 16;
                const uint32_t need = end > first ? (end - first + gstep - 1) / gstep : 0;
                if (known[c] < need) {
                    noc_async_read(get_noc_addr(w & 0xFF, (w >> 8) & 0xFF, prog_addr), scratch + 16 * c, 4);
                    ++pending;
                }
            }
            if (!pending) {
                return;
            }
            noc_async_read_barrier();
            invalidate_l1_cache();
            volatile tt_l1_ptr uint32_t* sp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
            for (uint32_t c = 0; c < n_gu; ++c) {
                const uint32_t w = get_arg_val<uint32_t>(gudn_rt + 1 + c);
                const uint32_t first = w >> 16;
                const uint32_t need = end > first ? (end - first + gstep - 1) / gstep : 0;
                if (known[c] < need) {
                    known[c] = sp[4 * c];
                }
            }
        }
    };
#endif
    for (uint32_t s0 = 0; s0 < slots; s0 += chunk) {
        const uint32_t cs = slots - s0 < chunk ? slots - s0 : chunk;
        cb_reserve_back(cb_in0, groups * cs * kt);
#ifdef GUDN
        // the remote words land in this chunk's cb_in0 space before its activations overwrite it (16 B per core)
        wait_slots(S0 + s0 + cs, get_write_ptr(cb_in0));
#endif
        const uint32_t in0 = get_write_ptr(cb_in0);
        for (uint32_t g = 0; g < groups; ++g) {
            for (uint32_t s = s0; s < s0 + cs; ++s) {
#ifdef ACT_BLK
                // the slot's K block of this row group: one page (gur_writer.cpp ACT_BLK)
                noc_async_read(s_act.get_noc_addr((S0 + s) * groups + g), in0 + (g * cs + s - s0) * kt * tb, kt * tb);
#else
                for (uint32_t k = 0; k < kt; ++k) {
                    noc_async_read(
                        s_act.get_noc_addr(((S0 + s) * kt + k) * groups + g), in0 + ((g * cs + s - s0) * kt + k) * tb, tb);
                }
#endif
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_in0, groups * cs * kt);
    }

    cb_wait_front(cb_out, groups * cols);
    const uint32_t src = get_read_ptr(cb_out);
    for (uint32_t g = 0; g < groups; ++g) {
        const uint32_t r0 = g * trows;
        const uint32_t roff = (r0 / 16) * 1024 + (r0 % 16) * 32;
        for (uint32_t c = 0; c < cols; ++c) {
            const uint32_t n = bank + banks * (c0 + c);
            const uint32_t t = src + (g * cols + c) * tb;
            noc_async_write(t, s_out.get_noc_addr(n, roff), half);
            noc_async_write(t + half, s_out.get_noc_addr(n, roff + 512), half);
        }
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, groups * cols);
    cb_pop_front(cb_tab, 1);
}
