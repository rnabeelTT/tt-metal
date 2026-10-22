// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Multi-row activation gather + SwiGLU output scatter (BRISC, NOC1) of the fused MoE decode (moe_decode/union.py
// ExpertGateUpRows), M decode rows on [trows, 32] tiles (trows = ROWS = 1 / 2 / 4 / 8 >= M; padding rows zero).
//
// 1. Reads rows 0..M-1 of the normed hidden x [32, H] (BF16 TILE): per tile, face 0 and face 1 rows 0..M-1 = one
//    [M, 32] activation tile (M x 64 bytes) of cb_x.
// 2. Reads the route table, builds the union (router_topk.hpp route_union), tells the three compute threads how many
//    union slots this core owns (TRISC mailboxes), and builds per slot the [M, 32] weight tile (row r = row r's
//    weight of the slot's expert, 0 when row r did not route to it; the shared slot: row r's shared-gate sigmoid),
//    rounded to the 7 significant bits custom_mm reads, followed by a zero tile (custom_mm takes an even K).
// 3. Writes each [M, 32] output tile (slot s, columns 32 j ..) to page s * pairs + j of the activation buffer.
//
// runtime args: [x_addr, act_addr, table_addr, pair j, slot0]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "router_topk.hpp"
#include "tools/profiler/kernel_profiler.hpp"

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
    constexpr uint32_t trows = get_compile_time_arg_val(9);  // tile rows (custom_mm: 1 / 2 / 4 / 8) >= rows
    constexpr auto x_args = TensorAccessorArgs<10>();
    constexpr auto act_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<act_args.next_compile_time_args_offset()>();

    constexpr uint32_t tb = trows * 64;  // one [trows, 32] BF16 tile
    constexpr uint32_t half = trows * 32;

    const auto s_x = TensorAccessor(x_args, x_addr, 2048);
    const auto s_act = TensorAccessor(act_args, act_addr, tb);
    const auto s_table = TensorAccessor(table_args, table_addr, 64);

    cb_reserve_back(cb_tab, 1);
    const uint32_t tab = get_write_ptr(cb_tab);
    for (uint32_t r = 0; r < rows; ++r) {
        noc_async_read(s_table.get_noc_addr(r), tab + 64 * r, 64);
    }
    cb_reserve_back(cb_x, kx);
    const uint32_t x_l1 = get_write_ptr(cb_x);
    for (uint32_t t = 0; t < kx; ++t) {
        noc_async_read(s_x.get_noc_addr(t, 0), x_l1 + t * tb, half);
        noc_async_read(s_x.get_noc_addr(t, 512), x_l1 + t * tb + half, half);
    }
    noc_async_read_barrier();
    if constexpr (trows > rows) {  // padding rows: zeros (their routing weights are 0, and 0 x NaN is not)
        for (uint32_t t = 0; t < kx; ++t) {
            for (uint32_t f = 0; f < 2; ++f) {
                volatile tt_l1_ptr uint32_t* p =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(x_l1 + t * tb + f * half + rows * 32);
                for (uint32_t i = 0; i < (trows - rows) * 8; ++i) {
                    p[i] = 0;
                }
            }
        }
    }
    cb_push_back(cb_x, kx);
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

    {
        DeviceZoneScopedN("wtiles");
        // every per-row weight tile of this core's slots (+ a zero tile each, custom_mm takes an even K): zeroed by
        // NOC reads of the L1 zero region, then one pass over the route records writes the rows that routed to a slot
        cb_reserve_back(cb_sc, 2 * mine);
        const uint32_t sc = get_write_ptr(cb_sc);
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t off = 0; off < 2 * mine * tb; off += MEM_ZEROS_SIZE) {
            const uint32_t n = 2 * mine * tb - off < MEM_ZEROS_SIZE ? 2 * mine * tb - off : MEM_ZEROS_SIZE;
            noc_async_read(zeros, sc + off, n);
        }
        noc_async_read_barrier();
        auto put = [&](uint32_t s, uint32_t r, uint32_t w) {
            if (s < slot0 || (s - slot0) % step) {
                return;
            }
            w = (w + 1) & 0xFFFE;  // 7 significant bits, half away from zero (the srcB width)
            const uint32_t w2 = w | (w << 16);
            // row r of a [trows, 32] tile: 16 values in face 0 (word 8 r ..), 16 in face 1 (word 8 (trows + r) ..)
            volatile tt_l1_ptr uint32_t* t32 =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sc + 2 * ((s - slot0) / step) * tb);
            for (uint32_t q = 0; q < 8; ++q) {
                t32[8 * r + q] = w2;
                t32[8 * (trows + r) + q] = w2;
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
        cb_push_back(cb_sc, 2 * mine);
    }

    for (uint32_t s = slot0; s < U + 1; s += step) {
        cb_wait_front(cb_act, 1);
        noc_async_write(get_read_ptr(cb_act), s_act.get_noc_addr(s * pairs + pair), tb);
        noc_async_writes_flushed();
        cb_pop_front(cb_act, 1);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_tab, 1);
}
