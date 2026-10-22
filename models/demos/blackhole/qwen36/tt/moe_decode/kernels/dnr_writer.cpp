// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Multi-row down-projection activation gather + output scatter (BRISC, NOC1) of the fused MoE decode
// (moe_decode/union.py ExpertDownRows), M rows on [trows, 32] tiles.
//
// 1. Builds the union from the route table (the slot count, to the three compute threads through the mailboxes).
// 2. in0 = the activation buffer's [M, 32] tiles of the U + 1 union slots, slot-major (pages s * kt .. s * kt + kt - 1):
//    the K order of the stacked down weights.
// 3. Writes each [M, 32] output tile (column n = bank + banks * (c0 + c)) into rows 0..M-1 of tile n of the [32, H]
//    BF16 TILE output (face 0 at byte 0, face 1 at byte 512; M x 32 bytes each).
//
// runtime args: [act_addr, out_addr, table_addr, bank, c0]

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
    constexpr uint32_t trows = get_compile_time_arg_val(8);  // tile rows (1 / 2 / 4 / 8) >= rows
    constexpr auto act_args = TensorAccessorArgs<9>();
    constexpr auto out_args = TensorAccessorArgs<act_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    constexpr uint32_t tb = trows * 64;
    constexpr uint32_t half = trows * 32;

    const auto s_act = TensorAccessor(act_args, act_addr, tb);
    const auto s_out = TensorAccessor(out_args, out_addr, 2048);
    const auto s_table = TensorAccessor(table_args, table_addr, 64);

    cb_reserve_back(cb_tab, 1);
    const uint32_t tab = get_write_ptr(cb_tab);
    for (uint32_t r = 0; r < rows; ++r) {
        noc_async_read(s_table.get_noc_addr(r), tab + 64 * r, 64);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    uint16_t uni[rows * topk];
    const uint32_t slots = 1 + moe_topk::route_union<topk>(reinterpret_cast<const uint16_t*>(tab), rows, uni);
    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, slots);
    ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, slots);
    ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, slots);

    const uint32_t kin = slots * kt;
    cb_reserve_back(cb_in0, kin);
    const uint32_t in0 = get_write_ptr(cb_in0);
    for (uint32_t p = 0; p < kin; ++p) {
        noc_async_read(s_act.get_noc_addr(p), in0 + p * tb, tb);
    }
    noc_async_read_barrier();
    cb_push_back(cb_in0, kin);

    cb_wait_front(cb_out, cols);
    const uint32_t src = get_read_ptr(cb_out);
    for (uint32_t c = 0; c < cols; ++c) {
        const uint32_t n = bank + banks * (c0 + c);
        noc_async_write(src + c * tb, s_out.get_noc_addr(n, 0), half);
        noc_async_write(src + c * tb + half, s_out.get_noc_addr(n, 512), half);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, cols);
    cb_pop_front(cb_tab, 1);
}
