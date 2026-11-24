// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Inkling short conv over R <= 29 rows, reader. Per 32-channel column c of this core: the input tile x (rows 0..R-1),
// the 3 carried rows (the fp32 row-major state [3, C]: 128-byte slices), and the 4 tap tiles. The sequence
// seq = [s0, s1, s2, x0 .. x_{R-1}] is laid out as shifted tiles T_j (row r = seq[r + j], j = 0, 1, 2) for the compute
// (T_3 is x itself). T_0 (the sequence) goes to the seq output for a later commit (commit.cpp); with commit_all the
// state is advanced past all R rows here (state rows = seq[R .. R + 2]).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "tile_rows.hpp"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t st_addr = get_arg_val<uint32_t>(1);
    const uint32_t taps_addr = get_arg_val<uint32_t>(2);
    const uint32_t seq_addr = get_arg_val<uint32_t>(3);
    const uint32_t c0 = get_arg_val<uint32_t>(4);
    const uint32_t n = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_s = get_compile_time_arg_val(1);
    constexpr uint32_t cb_t = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(3);
    constexpr uint32_t cols = get_compile_time_arg_val(4);
    constexpr uint32_t R = get_compile_time_arg_val(5);
    constexpr uint32_t commit_all = get_compile_time_arg_val(6);
    constexpr uint32_t TB = 2048;        // bf16 tile
    constexpr uint32_t ROWB = cols * 128;  // one fp32 state row (C channels)
    constexpr auto x_args = TensorAccessorArgs<7>();
    constexpr auto st_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto taps_args = TensorAccessorArgs<st_args.next_compile_time_args_offset()>();
    constexpr auto seq_args = TensorAccessorArgs<taps_args.next_compile_time_args_offset()>();

    const auto s_x = TensorAccessor(x_args, x_addr, TB);
    const auto s_st = TensorAccessor(st_args, st_addr, ROWB);
    const auto s_taps = TensorAccessor(taps_args, taps_addr, TB);
    const auto s_seq = TensorAccessor(seq_args, seq_addr, TB);

    const uint32_t scr = get_write_ptr(cb_scr);  // 3 state slices (384 B) + 3 new-state slices (384 B)
    volatile uint32_t* st32 = reinterpret_cast<volatile uint32_t*>(scr);
    volatile uint32_t* nst32 = reinterpret_cast<volatile uint32_t*>(scr + 384);

    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t c = c0 + i;
        cb_reserve_back(cb_x, 1);
        cb_reserve_back(cb_s, 3);
        cb_reserve_back(cb_t, 4);
        const uint32_t x = get_write_ptr(cb_x);
        const uint32_t s = get_write_ptr(cb_s);
        const uint32_t t = get_write_ptr(cb_t);
        noc_async_read(s_x.get_noc_addr(c), x, TB);
        for (uint32_t r = 0; r < 3; ++r) {
            noc_async_read(s_st.get_noc_addr(r, c * 128), scr + r * 128, 128);
        }
        for (uint32_t j = 0; j < 4; ++j) {
            noc_async_read(s_taps.get_noc_addr(j * cols + c), t + j * TB, TB);
        }
        noc_async_read_barrier();
        volatile uint32_t* xw = reinterpret_cast<volatile uint32_t*>(x);
        for (uint32_t j = 0; j < 3; ++j) {
            volatile uint32_t* tw = reinterpret_cast<volatile uint32_t*>(s + j * TB);
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t q = r + j;  // sequence index
                if (q < 3) {
                    sconv::f32_to_row(tw, r, st32 + q * 32);
                } else if (q - 3 < R) {
                    sconv::copy_row(tw, r, xw, q - 3);
                } else {
                    sconv::zero_row(tw, r);
                }
            }
        }
        if (seq_addr != 0) {
            noc_async_write(s, s_seq.get_noc_addr(c), TB);
        }
        if constexpr (commit_all) {
            // state := seq[R .. R + 2]
            for (uint32_t r = 0; r < 3; ++r) {
                const uint32_t q = R + r;
                if (q < 3) {
                    for (uint32_t w = 0; w < 32; ++w) {
                        nst32[r * 32 + w] = st32[q * 32 + w];
                    }
                } else {
                    sconv::row_to_f32(nst32 + r * 32, xw, q - 3);
                }
            }
            for (uint32_t r = 0; r < 3; ++r) {
                noc_async_write(scr + 384 + r * 128, s_st.get_noc_addr(r, c * 128), 128);
            }
        }
        noc_async_write_barrier();
        cb_push_back(cb_x, 1);
        cb_push_back(cb_s, 3);
        cb_push_back(cb_t, 4);
    }
}
