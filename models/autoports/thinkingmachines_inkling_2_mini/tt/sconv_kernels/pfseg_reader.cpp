// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Inkling short conv over a PACKED prefill chunk (pf_reader.cpp for several sequences): the chunk's R = Rt x 32 rows
// hold up to MAXSEG segments, each a different sequence with its own state. Segment s covers row tiles
// [ks[s], ke[s]); the rows before its first tile come from its state st[s] (not from the previous segment's rows). Two
// state points per segment are written out: n[s] (the commit: the state after its first n rows -> stn[s]) and, when
// snap[s] != NONE, snap[s] (a prefix-cache snapshot -> sb[s]). A state point P takes seq_s[P .. P + 2]
// (seq_s = [st_s rows, the segment's rows]). Same work split / compute / writer as the single-sequence kernel.
//
// common args: x_addr, taps_addr, nseg, then per segment (MAXSEG slots): ks, ke, st, stn, n, sb, snap

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "tile_rows.hpp"

constexpr uint32_t MAXSEG = 8;
constexpr uint32_t NONE = 0xffffffffu;

void kernel_main() {
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t taps_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t nseg = get_common_arg_val<uint32_t>(2);

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
    const auto s_taps = TensorAccessor(taps_args, taps_addr, TB);

    // scratch: carried rows (3 x 32 fp32), one output row (32 fp32), one bf16 tile (x_{k0-1})
    const uint32_t scr = get_write_ptr(cb_scr);
    volatile uint32_t* prev = reinterpret_cast<volatile uint32_t*>(scr);
    const uint32_t orow = scr + 384;
    volatile uint32_t* orw = reinterpret_cast<volatile uint32_t*>(orow);
    const uint32_t tmp = scr + 768;
    volatile uint32_t* tmpw = reinterpret_cast<volatile uint32_t*>(tmp);

    auto arg = [&](uint32_t s, uint32_t f) { return get_common_arg_val<uint32_t>(3 + s * 7 + f); };
    auto seg_of = [&](uint32_t k) {
        uint32_t s = 0;
        while (s + 1 < nseg && k >= arg(s, 1)) {
            ++s;
        }
        return s;
    };
    auto load_state = [&](uint32_t s) {
        const auto s_st = TensorAccessor(st_args, arg(s, 2), ROWB);
        for (uint32_t r = 0; r < 3; ++r) {
            noc_async_read(s_st.get_noc_addr(r, c * 128), scr + r * 128, 128);
        }
        noc_async_read_barrier();
    };
    // write state row r of point P of segment s (value in orw) to dst (stn or snapshot buffer)
    auto put = [&](uint32_t dst, uint32_t r) {
        const auto s_d = TensorAccessor(st_args, dst, ROWB);
        noc_async_write(orow, s_d.get_noc_addr(r, c * 128), 128);
        noc_async_write_barrier();
    };

    // the taps, once per core (the compute holds them for the whole range)
    cb_reserve_back(cb_t, 4);
    const uint32_t t = get_write_ptr(cb_t);
    for (uint32_t j = 0; j < 4; ++j) {
        noc_async_read(s_taps.get_noc_addr(j * cols + c), t + j * TB, TB);
    }
    noc_async_read_barrier();
    cb_push_back(cb_t, 4);
    if (k0 >= k1) {
        return;
    }
    // the rows before tile k0 (a segment start: its state, else the last rows of x_{k0-1})
    {
        const uint32_t s = seg_of(k0);
        if (k0 != arg(s, 0)) {
            noc_async_read(s_x.get_noc_addr((k0 - 1) * cols + c), tmp, TB);
            noc_async_read_barrier();
            for (uint32_t r = 0; r < 3; ++r) {
                sconv::row_to_f32(prev + r * 32, tmpw, 29 + r);
            }
        }
    }

    for (uint32_t k = k0; k < k1; ++k) {
        const uint32_t s = seg_of(k);
        const uint32_t ks = arg(s, 0);
        if (k == ks) {
            load_state(s);
            // state points inside the old state (P < 3): rows P + r < 3 come from it
            for (uint32_t pt = 0; pt < 2; ++pt) {
                const uint32_t P = pt == 0 ? arg(s, 4) : arg(s, 6);
                const uint32_t dst = pt == 0 ? arg(s, 3) : arg(s, 5);
                if (P == NONE) {
                    continue;
                }
                for (uint32_t r = 0; r < 3; ++r) {
                    if (P + r < 3) {
                        for (uint32_t w = 0; w < 32; ++w) {
                            orw[w] = prev[(P + r) * 32 + w];
                        }
                        put(dst, r);
                    }
                }
            }
        }
        cb_reserve_back(cb_x, 1);
        cb_reserve_back(cb_s, 3);
        const uint32_t x = get_write_ptr(cb_x);
        const uint32_t sp = get_write_ptr(cb_s);
        noc_async_read(s_x.get_noc_addr(k * cols + c), x, TB);
        noc_async_read_barrier();
        volatile uint32_t* xw = reinterpret_cast<volatile uint32_t*>(x);
        for (uint32_t j = 0; j < 3; ++j) {
            volatile uint32_t* tw = reinterpret_cast<volatile uint32_t*>(sp + j * TB);
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t q = r + j;  // index into [carried rows, x_k rows]
                if (q < 3) {
                    sconv::f32_to_row(tw, r, prev + q * 32);
                } else {
                    sconv::copy_row(tw, r, xw, q - 3);
                }
            }
        }
        // state rows that are rows of this tile: seq_s[P + r] = segment row P + r - 3
        for (uint32_t pt = 0; pt < 2; ++pt) {
            const uint32_t P = pt == 0 ? arg(s, 4) : arg(s, 6);
            const uint32_t dst = pt == 0 ? arg(s, 3) : arg(s, 5);
            if (P == NONE) {
                continue;
            }
            for (uint32_t r = 0; r < 3; ++r) {
                if (P + r >= 3) {
                    const uint32_t gr = ks * 32 + P + r - 3;
                    if (gr / 32 == k) {
                        sconv::row_to_f32(orw, xw, gr % 32);
                        put(dst, r);
                    }
                }
            }
        }
        for (uint32_t r = 0; r < 3; ++r) {
            sconv::row_to_f32(prev + r * 32, xw, 29 + r);
        }
        cb_push_back(cb_x, 1);
        cb_push_back(cb_s, 3);
    }
}
