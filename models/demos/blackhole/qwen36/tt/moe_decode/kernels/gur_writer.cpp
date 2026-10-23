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
// runtime args: [x_addr, act_addr, table_addr, group q, slot0]

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
    const uint32_t U = moe_topk::route_union<topk>(table, rows, uni);
    uint32_t mine = 0;
    for (uint32_t s = slot0; s < U + 1; s += step) {
        ++mine;
    }
    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, mine);
    ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, mine);
    ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, mine);

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
            const uint32_t e = table[32 * r + j];
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
    cb_pop_front(cb_wt, 1);
    cb_pop_front(cb_tab, 1);
}
