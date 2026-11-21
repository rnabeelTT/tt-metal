// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Inkling short conv over a prefill chunk (R = Rt x 32 rows), reader. Work item = (32-channel column c, a range of
// row tiles [k0, k1)); the core's item comes from its logical coordinates (index x * GY + y; item i: c = i % cols,
// group g = i / cols), so every argument is common to all cores. Per row tile k: the input tile x_k and the shifted
// tiles T_j (row r = seq[32k + r + j], seq = [s0, s1, s2, x_0 ..], j = 0, 1, 2), the rows before the tile coming from
// the carried rows (the state for k = 0, else the last 3 rows of x_{k-1}). The new state (seq[n .. n + 2], n = the
// chunk's valid rows) is written to the separate stn tensor by the cores that hold those rows (no core reads what
// another writes); the host copies stn into the state after the program.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "tile_rows.hpp"

void kernel_main() {
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t st_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t taps_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t stn_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t n_commit = get_common_arg_val<uint32_t>(4);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_s = get_compile_time_arg_val(1);
    constexpr uint32_t cb_t = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(3);
    constexpr uint32_t cols = get_compile_time_arg_val(4);
    constexpr uint32_t Rt = get_compile_time_arg_val(5);
    constexpr uint32_t G = get_compile_time_arg_val(6);
    constexpr uint32_t GY = get_compile_time_arg_val(7);
    constexpr uint32_t TB = 2048;          // bf16 tile
    constexpr uint32_t ROWB = cols * 128;  // one fp32 state row (C channels)
    constexpr auto x_args = TensorAccessorArgs<8>();
    constexpr auto st_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto taps_args = TensorAccessorArgs<st_args.next_compile_time_args_offset()>();
    constexpr auto stn_args = TensorAccessorArgs<taps_args.next_compile_time_args_offset()>();

    const uint32_t item = (uint32_t)get_absolute_logical_x() * GY + (uint32_t)get_absolute_logical_y();
    if (item >= cols * G) {
        return;
    }
    const uint32_t c = item % cols;
    const uint32_t g = item / cols;
    constexpr uint32_t per = (Rt + G - 1) / G;
    const uint32_t k0 = g * per;
    const uint32_t k1 = (k0 + per < Rt) ? k0 + per : Rt;

    const auto s_x = TensorAccessor(x_args, x_addr, TB);
    const auto s_st = TensorAccessor(st_args, st_addr, ROWB);
    const auto s_taps = TensorAccessor(taps_args, taps_addr, TB);
    const auto s_stn = TensorAccessor(stn_args, stn_addr, ROWB);

    // scratch: carried rows (3 x 32 fp32), new-state rows (3 x 32 fp32), one bf16 tile (x_{k0-1})
    const uint32_t scr = get_write_ptr(cb_scr);
    volatile uint32_t* prev = reinterpret_cast<volatile uint32_t*>(scr);
    volatile uint32_t* nst = reinterpret_cast<volatile uint32_t*>(scr + 384);
    const uint32_t tmp = scr + 768;
    volatile uint32_t* tmpw = reinterpret_cast<volatile uint32_t*>(tmp);

    // the taps, once per core (the compute holds them for the whole range)
    cb_reserve_back(cb_t, 4);
    const uint32_t t = get_write_ptr(cb_t);
    for (uint32_t j = 0; j < 4; ++j) {
        noc_async_read(s_taps.get_noc_addr(j * cols + c), t + j * TB, TB);
    }
    // the rows before tile k0
    if (k0 < k1) {
        if (k0 == 0) {
            for (uint32_t r = 0; r < 3; ++r) {
                noc_async_read(s_st.get_noc_addr(r, c * 128), scr + r * 128, 128);
            }
            noc_async_read_barrier();
        } else {
            noc_async_read(s_x.get_noc_addr((k0 - 1) * cols + c), tmp, TB);
            noc_async_read_barrier();
            for (uint32_t r = 0; r < 3; ++r) {
                sconv::row_to_f32(prev + r * 32, tmpw, 29 + r);
            }
        }
    } else {
        noc_async_read_barrier();
    }
    cb_push_back(cb_t, 4);

    // new state rows owned here: row r = seq[n + r] = x[n - 3 + r] (or the old state row n + r when n + r < 3)
    bool own[3];
    for (uint32_t r = 0; r < 3; ++r) {
        const int32_t gr = (int32_t)n_commit - 3 + (int32_t)r;
        if (gr < 0) {
            own[r] = (k0 == 0 && k0 < k1);
            if (own[r]) {
                for (uint32_t w = 0; w < 32; ++w) {
                    nst[r * 32 + w] = prev[(n_commit + r) * 32 + w];
                }
            }
        } else {
            const uint32_t kt = (uint32_t)gr / 32;
            own[r] = (kt >= k0 && kt < k1);
        }
    }

    for (uint32_t k = k0; k < k1; ++k) {
        cb_reserve_back(cb_x, 1);
        cb_reserve_back(cb_s, 3);
        const uint32_t x = get_write_ptr(cb_x);
        const uint32_t s = get_write_ptr(cb_s);
        noc_async_read(s_x.get_noc_addr(k * cols + c), x, TB);
        noc_async_read_barrier();
        volatile uint32_t* xw = reinterpret_cast<volatile uint32_t*>(x);
        for (uint32_t j = 0; j < 3; ++j) {
            volatile uint32_t* tw = reinterpret_cast<volatile uint32_t*>(s + j * TB);
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t q = r + j;  // index into [carried rows, x_k rows]
                if (q < 3) {
                    sconv::f32_to_row(tw, r, prev + q * 32);
                } else {
                    sconv::copy_row(tw, r, xw, q - 3);
                }
            }
        }
        for (uint32_t r = 0; r < 3; ++r) {
            const int32_t gr = (int32_t)n_commit - 3 + (int32_t)r;
            if (own[r] && gr >= 0 && (uint32_t)gr / 32 == k) {
                sconv::row_to_f32(nst + r * 32, xw, (uint32_t)gr % 32);
            }
        }
        for (uint32_t r = 0; r < 3; ++r) {
            sconv::row_to_f32(prev + r * 32, xw, 29 + r);
        }
        cb_push_back(cb_x, 1);
        cb_push_back(cb_s, 3);
    }
    for (uint32_t r = 0; r < 3; ++r) {
        if (own[r]) {
            noc_async_write(scr + 384 + r * 128, s_stn.get_noc_addr(r, c * 128), 128);
        }
    }
    noc_async_write_barrier();
}
