// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// GDN decode short conv (K = 4), reader: per qkv tile column c of this core, the three carried input rows h0..h2
// (conv_states[1..3], [Bmax <= 32, qkv] bf16 TILE: one tile row, page c), this step's input x (column c of the in-proj
// output, [B <= 32, P] bf16 TILE) and the column's 4 tap tiles (taps [4, 1, 32, qkv]: every row holds the tap row).
// It also shifts the register in place (conv_states[1..3] <- h1, h2, x: every user of column c lives in that one tile,
// so no other core touches it) before handing the tiles to the compute.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t proj_addr = get_arg_val<uint32_t>(0);
    const uint32_t st1_addr = get_arg_val<uint32_t>(1);
    const uint32_t st2_addr = get_arg_val<uint32_t>(2);
    const uint32_t st3_addr = get_arg_val<uint32_t>(3);
    const uint32_t taps_addr = get_arg_val<uint32_t>(4);
    const uint32_t c0 = get_arg_val<uint32_t>(5);
    const uint32_t n = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_h = get_compile_time_arg_val(1);
    constexpr uint32_t cb_t = get_compile_time_arg_val(2);
    constexpr uint32_t cols = get_compile_time_arg_val(3);  // qkv tiles
    constexpr uint32_t TB = 2048;                            // bf16 tile
    constexpr auto proj_args = TensorAccessorArgs<4>();
    constexpr auto st_args = TensorAccessorArgs<proj_args.next_compile_time_args_offset()>();
    constexpr auto taps_args = TensorAccessorArgs<st_args.next_compile_time_args_offset()>();

    const auto s_proj = TensorAccessor(proj_args, proj_addr, TB);
    const auto s_st1 = TensorAccessor(st_args, st1_addr, TB);
    const auto s_st2 = TensorAccessor(st_args, st2_addr, TB);
    const auto s_st3 = TensorAccessor(st_args, st3_addr, TB);
    const auto s_taps = TensorAccessor(taps_args, taps_addr, TB);

    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t c = c0 + i;
        cb_reserve_back(cb_h, 3);
        cb_reserve_back(cb_x, 1);
        cb_reserve_back(cb_t, 4);
        const uint32_t h = get_write_ptr(cb_h);
        const uint32_t x = get_write_ptr(cb_x);
        const uint32_t t = get_write_ptr(cb_t);
        noc_async_read(s_st1.get_noc_addr(c), h, TB);
        noc_async_read(s_st2.get_noc_addr(c), h + TB, TB);
        noc_async_read(s_st3.get_noc_addr(c), h + 2 * TB, TB);
        noc_async_read(s_proj.get_noc_addr(c), x, TB);
        for (uint32_t j = 0; j < 4; ++j) {
            noc_async_read(s_taps.get_noc_addr(j * cols + c), t + j * TB, TB);
        }
        noc_async_read_barrier();
        // shift: the register's new rows 1..3 are the old rows 2, 3 and this step's input
        noc_async_write(h + TB, s_st1.get_noc_addr(c), TB);
        noc_async_write(h + 2 * TB, s_st2.get_noc_addr(c), TB);
        noc_async_write(x, s_st3.get_noc_addr(c), TB);
        noc_async_write_barrier();  // the slots are recycled once the compute pops them
        cb_push_back(cb_h, 3);
        cb_push_back(cb_x, 1);
        cb_push_back(cb_t, 4);
    }
}
