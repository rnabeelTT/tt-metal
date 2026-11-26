// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Multi-row down-projection activation gather + output scatter (BRISC, NOC1) of the fused MoE decode
// (moe_decode/union.py ExpertDownRows): M rows as `groups` row groups of [trows, 32] tiles.
// This port's copy (tt/arpush.py): with ARP_T the output goes into the fused all-reduce instead (tt/ccl.py push): each
// column's ARP_T live rows straight into this die's slot on the shard cores and the link senders' payload, packed as
// arpack.cpp (tile n's chunk at 64 T n: face 0's T rows, then face 1's), then one increment per touched shard core /
// sender.
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
// runtime args: [act_addr, out_addr, table_addr, bank, c0, (EP_EXPERTS) ep_offset_addr]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "models/demos/blackhole/qwen36/tt/moe_decode/kernels/router_topk.hpp"
#include "models/demos/blackhole/qwen36/tt/moe_decode/kernels/moe_zones.hpp"

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
    const auto s_table = TensorAccessor(table_args, table_addr, 64);

    cb_reserve_back(cb_tab, 1);
    const uint32_t tab = get_write_ptr(cb_tab);
    for (uint32_t r = 0; r < rows; ++r) {
        noc_async_read(s_table.get_noc_addr(r), tab + 64 * r, 64);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    uint16_t uni[rows * topk * MOE_SUBS];
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
    const uint32_t slots = 1 - S0 + moe_topk::expand_subs<MOE_SUBS>(
        uni, moe_topk::route_union<topk>(reinterpret_cast<const uint16_t*>(tab), rows, uni));
#endif
    MOE_MARK("D_WU");
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

    for (uint32_t s0 = 0; s0 < slots; s0 += chunk) {
        const uint32_t cs = slots - s0 < chunk ? slots - s0 : chunk;
        cb_reserve_back(cb_in0, groups * cs * kt);
        const uint32_t in0 = get_write_ptr(cb_in0);
        for (uint32_t g = 0; g < groups; ++g) {
            for (uint32_t s = s0; s < s0 + cs; ++s) {
                for (uint32_t k = 0; k < kt; ++k) {
                    noc_async_read(
                        s_act.get_noc_addr(((S0 + s) * kt + k) * groups + g), in0 + ((g * cs + s - s0) * kt + k) * tb, tb);
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_in0, groups * cs * kt);
        MOE_MARK("D_WA");
    }

    cb_wait_front(cb_out, groups * cols);
    MOE_MARK("D_WO");
    const uint32_t src = get_read_ptr(cb_out);
#ifdef ARP_T
    // push args (after the base ones): slot, K, shard noc (x, y) x K, sender noc (x, y) x links
    constexpr uint32_t PA = 5;
    const uint32_t slot = get_arg_val<uint32_t>(PA);
    const uint32_t K = get_arg_val<uint32_t>(PA + 1);
    auto shard_noc = [&](uint32_t k, uint32_t addr) {
        return get_noc_addr(get_arg_val<uint32_t>(PA + 2 + 2 * k), get_arg_val<uint32_t>(PA + 3 + 2 * k), addr);
    };
    auto send_noc = [&](uint32_t l, uint32_t addr) {
        return get_noc_addr(
            get_arg_val<uint32_t>(PA + 2 + 2 * K + 2 * l), get_arg_val<uint32_t>(PA + 3 + 2 * K + 2 * l), addr);
    };
    const uint32_t tx = get_write_ptr(ARP_CB_TX);
    auto put = [&](uint32_t s, uint32_t b, uint32_t n) {  // (a run may cross a shard tile)
        while (n > 0) {
            const uint32_t k = b / 2048, o = b % 2048, m = 2048 - o < n ? 2048 - o : n;
            noc_async_write(s, shard_noc(k, slot + o), m);
            noc_async_write(s, send_noc(k / ARP_KPL, tx + (k % ARP_KPL) * 2048 + o), m);
            b += m;
            s += m;
            n -= m;
        }
    };
    for (uint32_t c = 0; c < cols; ++c) {
        const uint32_t n = bank + banks * (c0 + c);
        const uint32_t t = src + c * tb;
        put(t, 64 * ARP_T * n, 32 * ARP_T);
        put(t + half, 64 * ARP_T * n + 32 * ARP_T, 32 * ARP_T);
    }
    noc_async_write_barrier();
    uint32_t lk = 0xffffffff, ll = 0xffffffff;
    for (uint32_t c = 0; c < cols; ++c) {  // (ascending n: the touched shard tiles / links ascend)
        const uint32_t b = 64 * ARP_T * (bank + banks * (c0 + c));
        for (uint32_t k = b / 2048; k <= (b + 64 * ARP_T - 1) / 2048; ++k) {
            if (k != lk) {
                noc_semaphore_inc(shard_noc(k, get_semaphore(ARP_SEM_SH)), 1);
                lk = k;
            }
            if (k / ARP_KPL != ll) {
                ll = k / ARP_KPL;
                noc_semaphore_inc(send_noc(ll, get_semaphore(ARP_SEM_TX)), 1);
            }
        }
    }
    noc_async_atomic_barrier();
#else
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
    MOE_MARK("D_WE");
    noc_async_write_barrier();
#endif
    cb_pop_front(cb_out, groups * cols);
    cb_pop_front(cb_tab, 1);
}
