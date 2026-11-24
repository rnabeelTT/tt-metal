// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Attention output gate (tt/agate.py), reader: core j (one output tile column of [1, 1, B, NH * HD]) gathers its
// live rows: o[b, h = j / DT, d tile j % DT] (the SDPA output [1, B, NH, HD], L1: row h of tile (b, j % DT)) into row
// b of cb_o, and rows b < B of the gate tile (column G0 + j of the fp32 q-projection output) into cb_g.
//
// common args: o, pq

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_o = get_compile_time_arg_val(0);
    constexpr uint32_t cb_g = get_compile_time_arg_val(1);
    constexpr uint32_t DT = get_compile_time_arg_val(2);  // HD / 32
    constexpr uint32_t G0 = get_compile_time_arg_val(3);  // the gate's first tile column in pq
    constexpr uint32_t CX = get_compile_time_arg_val(4);
    constexpr uint32_t B = AG_ROWS;
    constexpr auto o_args = TensorAccessorArgs<5>();
    constexpr auto g_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    const uint32_t core = (uint32_t)get_absolute_logical_y() * CX + (uint32_t)get_absolute_logical_x();
    const auto ao = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0), 2048);
    const auto ag = TensorAccessor(g_args, get_common_arg_val<uint32_t>(1), 4096);
    for (uint32_t jj = 0; jj < AG_J; ++jj) {  // this core's output tile columns
        const uint32_t j = core * AG_J + jj;
        const uint32_t h = j / DT, dt = j % DT;
        cb_reserve_back(cb_o, 1);
        cb_reserve_back(cb_g, 1);
        const uint32_t lo = get_write_ptr(cb_o), lg = get_write_ptr(cb_g);
        const uint32_t hoff = h < 16 ? h * 32 : 1024 + (h - 16) * 32;  // (head rows 16.. sit in faces 2 / 3)
        for (uint32_t b = 0; b < B; ++b) {
            const uint64_t src = ao.get_noc_addr(b * DT + dt) + hoff;
            noc_async_read(src, lo + b * 32, 32);
            noc_async_read(src + 512, lo + 512 + b * 32, 32);
        }
        const uint64_t gs = ag.get_noc_addr(G0 + j);
        noc_async_read(gs, lg, B * 64);
        noc_async_read(gs + 1024, lg + 1024, B * 64);
        noc_async_read_barrier();
        cb_push_back(cb_o, 1);
        cb_push_back(cb_g, 1);
    }
}
