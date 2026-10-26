// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Multi-row activation gather + SwiGLU output scatter (BRISC, NOC1) of the fused MoE decode (moe_decode/union.py
// ExpertGateUpRows): M decode rows as `groups` row groups of [trows, 32] tiles (trows = ROWS = 1 / 2 / 4 / 8; rows
// past M are zero).
//
// 1. Reads rows 0..M-1 of the normed hidden x [32, H] (BF16 TILE): per tile and row group g, the group's rows of face 0
//    and face 1 (row r of a 32x32 tile: byte (r / 16) * 1024 + (r % 16) * 32, + 512 for face 1) = one [trows, 32]
//    activation tile (cb_x tile g * kx + t).
// 2. Reads the route table, builds the union (router_topk.hpp route_union), tells the three compute threads how many
//    union slots this core owns (TRISC mailboxes), and fills this core's [slots, M] weight table in one pass over the
//    route records (row r's weight of the slot's expert, 0 when row r did not route to it; the shared slot: row r's
//    shared-gate sigmoid), rounded to the 7 significant bits custom_mm reads.
// 3. Per slot: the groups' [trows, 32] weight tiles (each followed by a zero tile: custom_mm takes an even K; zeroed by
//    NOC reads of the L1 zero region), then the previous slot's output tiles to the activation buffer (page
//    (s * pairs + p) * groups + g for each column pair p of the core's group, gu_layout.hpp): interleaved so neither
//    cb fills up.
//
// EP_EXPERTS: only this die's window of the union (local ids); other dies' experts get no weight.
//
// SPLIT_PAIR (m = 1; runtime args + [partner noc x, y]): gate and up columns on different cores, paired on the same slot
// sequence. ROLE_GATE writers fill cb_sc with the weight tiles (as above) and send each slot's g' = w * silu(g) tiles
// into the paired up core's cb_sc ring (same slot positions: the CBs sit at the same address on every core), after a
// credit (program semaphore 0 here, granted by the up writer once the ring slot is free and zeroed), then increment
// the partner's semaphore 1. ROLE_UP writers zero their cb_sc ring, grant credits, publish each slot once its data
// arrived (semaphore 1), and write the activation buffer (pair q) as above.
//
// runtime args: [x_addr, act_addr, table_addr, group q, slot0, (EP_EXPERTS) ep_offset_addr]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "gu_layout.hpp"
#include "router_topk.hpp"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t act_addr = get_arg_val<uint32_t>(1);
    const uint32_t table_addr = get_arg_val<uint32_t>(2);
    const uint32_t pair = get_arg_val<uint32_t>(3);
    const uint32_t slot0 = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_act = get_compile_time_arg_val(1);
    constexpr uint32_t cb_sc = get_compile_time_arg_val(2);
    constexpr uint32_t cb_tab = get_compile_time_arg_val(3);
    constexpr uint32_t kx = get_compile_time_arg_val(4);
    constexpr uint32_t topk = get_compile_time_arg_val(5);
    constexpr uint32_t rows = get_compile_time_arg_val(6);
    constexpr uint32_t step = get_compile_time_arg_val(7);
    constexpr uint32_t pairs = get_compile_time_arg_val(8);
    constexpr uint32_t trows = get_compile_time_arg_val(9);  // tile rows (custom_mm: 1 / 2 / 4 / 8)
    constexpr uint32_t groups = get_compile_time_arg_val(10);
    constexpr uint32_t cb_wt = get_compile_time_arg_val(11);
    constexpr uint32_t act_page = get_compile_time_arg_val(12);  // activation buffer page (>= one tile: shared by all widths)
    constexpr uint32_t m = get_compile_time_arg_val(13);  // up|gate columns per bank and expert
    constexpr uint32_t banks = get_compile_time_arg_val(14);
    constexpr uint32_t np = gu_layout::pairs<m>();
    constexpr auto x_args = TensorAccessorArgs<15>();
    constexpr auto act_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<act_args.next_compile_time_args_offset()>();

    constexpr uint32_t tb = trows * 64;  // one [trows, 32] BF16 tile
    constexpr uint32_t half = trows * 32;
    static_assert(groups * trows >= rows && (groups == 1 || trows == 8), "row groups of 8");

    const auto s_x = TensorAccessor(x_args, x_addr, 2048);
    const auto s_act = TensorAccessor(act_args, act_addr, act_page);
    const auto s_table = TensorAccessor(table_args, table_addr, 64);
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    auto zero_l1 = [&](uint32_t addr, uint32_t bytes) {
        for (uint32_t off = 0; off < bytes; off += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, addr + off, bytes - off < MEM_ZEROS_SIZE ? bytes - off : MEM_ZEROS_SIZE);
        }
    };

    cb_reserve_back(cb_tab, 1);
    const uint32_t tab = get_write_ptr(cb_tab);
    for (uint32_t r = 0; r < rows; ++r) {
        noc_async_read(s_table.get_noc_addr(r), tab + 64 * r, 64);
    }
    cb_reserve_back(cb_x, groups * kx);
    const uint32_t x_l1 = get_write_ptr(cb_x);
    for (uint32_t g = 0; g < groups; ++g) {
        const uint32_t r0 = g * trows;
        const uint32_t roff = (r0 / 16) * 1024 + (r0 % 16) * 32;
        for (uint32_t t = 0; t < kx; ++t) {
            const uint32_t dst = x_l1 + (g * kx + t) * tb;
            noc_async_read(s_x.get_noc_addr(t, roff), dst, half);
            noc_async_read(s_x.get_noc_addr(t, roff + 512), dst + half, half);
        }
    }
    noc_async_read_barrier();
    if constexpr (groups * trows > rows) {  // padding rows: zeros (their routing weights are 0, and 0 x NaN is not)
        constexpr uint32_t g = groups - 1;
        constexpr uint32_t first = rows - g * trows;
        for (uint32_t t = 0; t < kx; ++t) {
            for (uint32_t f = 0; f < 2; ++f) {
                volatile tt_l1_ptr uint32_t* p =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(x_l1 + (g * kx + t) * tb + f * half + first * 32);
                for (uint32_t i = 0; i < (trows - first) * 8; ++i) {
                    p[i] = 0;
                }
            }
        }
    }
    cb_push_back(cb_x, groups * kx);
    invalidate_l1_cache();

    const uint16_t* table = reinterpret_cast<const uint16_t*>(tab);
    uint16_t uni[rows * topk];
#ifdef EP_EXPERTS
    const uint32_t ep_off = moe_topk::read_ep_offset(get_arg_val<uint32_t>(5), tab + 64 * rows);
    const uint32_t U = moe_topk::window_union(uni, moe_topk::route_union<topk>(table, rows, uni), ep_off, EP_EXPERTS);
#else
    const uint32_t U = moe_topk::route_union<topk>(table, rows, uni);
#endif
    uint32_t mine = 0;
    for (uint32_t s = slot0; s < U + 1; s += step) {
        ++mine;
    }
    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, mine);
    ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, mine);
    ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, mine);

#ifdef SPLIT_PAIR
    constexpr uint32_t ring = SPLIT_RING;  // cb_sc holds `ring` slots of 2 * groups tiles
    constexpr uint32_t slot_bytes = 2 * groups * tb;
#ifdef EP_EXPERTS
    constexpr uint32_t partner_arg = 6;
#else
    constexpr uint32_t partner_arg = 5;
#endif
    const uint32_t px = get_arg_val<uint32_t>(partner_arg);
    const uint32_t py = get_arg_val<uint32_t>(partner_arg + 1);
#endif
#if defined(SPLIT_PAIR) && !defined(ROLE_GATE)
    // up core: the partner writes g' tiles into this cb_sc ring; zero it, grant `ring` credits, then per slot publish
    // once the data is in (and grant the slot two ahead when it is free again)
    {
        cb_reserve_back(cb_sc, ring * 2 * groups);
        zero_l1(get_write_ptr(cb_sc), ring * slot_bytes);
        noc_async_read_barrier();
        noc_semaphore_inc(get_noc_addr(px, py, get_semaphore(0)), ring);
        volatile tt_l1_ptr uint32_t* got = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
        auto drain_up = [&](uint32_t s) {
            for (uint32_t g = 0; g < groups; ++g) {
                cb_wait_front(cb_act, 1);
                noc_async_write(get_read_ptr(cb_act), s_act.get_noc_addr((s * pairs + pair) * groups + g), tb);
                noc_async_writes_flushed();
                cb_pop_front(cb_act, 1);
            }
        };
        uint32_t i = 0;
        for (uint32_t s = slot0; s < U + 1; s += step, ++i) {
            if (i >= ring) {
                cb_reserve_back(cb_sc, 2 * groups);  // slot i - ring consumed
                zero_l1(get_write_ptr(cb_sc), slot_bytes);
                noc_async_read_barrier();
                noc_semaphore_inc(get_noc_addr(px, py, get_semaphore(0)), 1);
            }
            noc_semaphore_wait_min(got, i + 1);
            cb_push_back(cb_sc, 2 * groups);
            if (s != slot0) {
                drain_up(s - step);
            }
        }
        if (mine) {
            drain_up(slot0 + (mine - 1) * step);
        }
        noc_async_write_barrier();
        noc_async_atomic_barrier();
        cb_pop_front(cb_tab, 1);
        return;
    }
#endif

    // this core's [mine, rows] weight table (BF16 bits, rounded to 7 significant bits), one pass over the records
    cb_reserve_back(cb_wt, 1);
    volatile tt_l1_ptr uint16_t* wt = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_wt));
    zero_l1(get_write_ptr(cb_wt), ((2 * mine * rows + 15) / 16) * 16);
    noc_async_read_barrier();
    auto put = [&](uint32_t s, uint32_t r, uint32_t w) {
        if (s >= slot0 && (s - slot0) % step == 0) {
            wt[((s - slot0) / step) * rows + r] = static_cast<uint16_t>((w + 1) & 0xFFFE);  // srcB width, half up
        }
    };
    for (uint32_t r = 0; r < rows; ++r) {
        put(0, r, table[32 * r + 16]);  // shared expert
        for (uint32_t j = 0; j < topk; ++j) {
#ifdef EP_EXPERTS
            if (table[32 * r + j] - ep_off >= EP_EXPERTS) {
                continue;  // another die's expert
            }
            const uint32_t e = table[32 * r + j] - ep_off;
#else
            const uint32_t e = table[32 * r + j];
#endif
            uint32_t lo = 0, hi = U;  // uni is ascending
            while (lo < hi) {
                const uint32_t mid = (lo + hi) / 2;
                if (uni[mid] < e) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            put(1 + lo, r, table[32 * r + 17 + j]);
        }
    }

#if defined(SPLIT_PAIR) && defined(ROLE_GATE)
    const uint32_t ring_base = get_write_ptr(cb_sc);  // cb_sc's start: the partner's ring sits at the same address
    volatile tt_l1_ptr uint32_t* credit = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    uint32_t sent = 0;
    auto drain = [&](uint32_t s) {
        (void)s;
        noc_semaphore_wait_min(credit, sent + 1);
        const uint32_t dst = ring_base + (sent % ring) * slot_bytes;
        for (uint32_t g = 0; g < groups; ++g) {
            cb_wait_front(cb_act, 1);
            noc_async_write(get_read_ptr(cb_act), get_noc_addr(px, py, dst + 2 * g * tb), tb);
            noc_async_writes_flushed();
            cb_pop_front(cb_act, 1);
        }
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(px, py, get_semaphore(1)), 1);
        ++sent;
    };
#else
    auto drain = [&](uint32_t s) {
        for (uint32_t g = 0; g < groups; ++g) {
            cb_wait_front(cb_act, np);
            for (uint32_t p = 0; p < np; ++p) {
                const uint32_t pj = gu_layout::pair_index<m, banks>(pair, p);
                noc_async_write(get_read_ptr(cb_act) + p * tb, s_act.get_noc_addr((s * pairs + pj) * groups + g), tb);
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_act, np);
        }
    };
#endif
    uint32_t i = 0;
    for (uint32_t s = slot0; s < U + 1; s += step, ++i) {
        cb_reserve_back(cb_sc, 2 * groups);
        const uint32_t sc = get_write_ptr(cb_sc);
        zero_l1(sc, 2 * groups * tb);
        noc_async_read_barrier();
        for (uint32_t g = 0; g < groups; ++g) {
            // row rr of a [trows, 32] tile: 16 values in face 0 (word 8 rr ..), 16 in face 1 (word 8 (trows + rr) ..)
            volatile tt_l1_ptr uint32_t* t32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sc + 2 * g * tb);
            for (uint32_t rr = 0; rr < trows && g * trows + rr < rows; ++rr) {
                const uint32_t w = wt[i * rows + g * trows + rr];
                if (w) {
                    const uint32_t w2 = w | (w << 16);
                    for (uint32_t q = 0; q < 8; ++q) {
                        t32[8 * rr + q] = w2;
                        t32[8 * (trows + rr) + q] = w2;
                    }
                }
            }
        }
        cb_push_back(cb_sc, 2 * groups);
        if (s != slot0) {
            drain(s - step);
        }
    }
    if (mine) {
        drain(slot0 + (mine - 1) * step);
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    cb_pop_front(cb_wt, 1);
    cb_pop_front(cb_tab, 1);
}
