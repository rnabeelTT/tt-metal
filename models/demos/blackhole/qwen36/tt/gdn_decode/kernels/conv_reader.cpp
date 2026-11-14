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
#ifdef ROWS
    // ROWS + BASE (DFlash, no commit): runtime args 7 / 8 = the base index tensor (uint32 word 0: the previous pass's
    // accepted row a, or 0xFFFFFFFF = take the carried rows from st1..st3) and the history tensor (per column c one
    // tile: row i = u[i - 3] of the previous pass, i.e. its three carried rows then its input rows). With a base, the
    // carried rows are history rows a + 1 .. a + 3; this pass's history (the h_0 tile below) replaces it afterwards.
    constexpr auto base_args = TensorAccessorArgs<taps_args.next_compile_time_args_offset()>();
    constexpr auto hist_args = TensorAccessorArgs<base_args.next_compile_time_args_offset()>();
    const auto s_base = TensorAccessor(base_args, get_arg_val<uint32_t>(7), 128);
    const auto s_hist = TensorAccessor(hist_args, get_arg_val<uint32_t>(8), TB);
    uint32_t base = 0xFFFFFFFFu;
    {
        const uint32_t scr = get_write_ptr(cb_t);  // scratch before the first tap read
        noc_async_read(s_base.get_noc_addr(0), scr, 64);
        noc_async_read_barrier();
        invalidate_l1_cache();
        base = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr);
    }
#endif

    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t c = c0 + i;
        cb_reserve_back(cb_h, 3);
        cb_reserve_back(cb_x, 1);
        cb_reserve_back(cb_t, 4);
        const uint32_t h = get_write_ptr(cb_h);
        const uint32_t x = get_write_ptr(cb_x);
        const uint32_t t = get_write_ptr(cb_t);
#ifdef ROWS
        if (base != 0xFFFFFFFFu) {
            noc_async_read(s_hist.get_noc_addr(c), h, TB);  // rows base + 1 .. base + 3 = the carried rows
        } else {
            noc_async_read(s_st1.get_noc_addr(c), h, TB);
            noc_async_read(s_st2.get_noc_addr(c), h + TB, TB);
            noc_async_read(s_st3.get_noc_addr(c), h + 2 * TB, TB);
        }
#else
        noc_async_read(s_st1.get_noc_addr(c), h, TB);
        noc_async_read(s_st2.get_noc_addr(c), h + TB, TB);
        noc_async_read(s_st3.get_noc_addr(c), h + 2 * TB, TB);
#endif
        noc_async_read(s_proj.get_noc_addr(c), x, TB);
        for (uint32_t j = 0; j < 4; ++j) {
            noc_async_read(s_taps.get_noc_addr(j * cols + c), t + j * TB, TB);
        }
        noc_async_read_barrier();
#ifdef ROWS
        // ROWS: the tile rows are consecutive positions of ONE user (row r = position p + r), its carried inputs in row 0
        // of st1..st3 (oldest first). The compute's h_j tile (tap j, j = 0..2) must hold u[r - 3 + j] in row r, where
        // u[-3..-1] = the carried rows and u[0..] = this step's x rows: built here by 32-byte face-row copies (no
        // register shift: the commit sets the carry from the accepted rows).
        invalidate_l1_cache();
        auto row_ptr = [](uint32_t base, uint32_t r, uint32_t half) {
            return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + ((r >> 4) * 2 + half) * 512 + (r & 15) * 32);
        };
        // carried rows (row 0 of each st tile) to a scratch copy first: the h tiles are rewritten in place
        uint32_t carry[3][16];
        for (uint32_t j = 0; j < 3; ++j) {
            for (uint32_t half = 0; half < 2; ++half) {
                // st_j row 0, or (base) history row base + 1 + j of the tile read into h_0
                volatile tt_l1_ptr uint32_t* src =
                    base != 0xFFFFFFFFu ? row_ptr(h, base + 1 + j, half) : row_ptr(h + j * TB, 0, half);
                for (uint32_t w = 0; w < 8; ++w) {
                    carry[j][half * 8 + w] = src[w];
                }
            }
        }
        for (uint32_t j = 0; j < 3; ++j) {      // h_j: shift s = 3 - j
            const uint32_t s = 3 - j;
            for (uint32_t r = 0; r < 32; ++r) {
                for (uint32_t half = 0; half < 2; ++half) {
                    volatile tt_l1_ptr uint32_t* dst = row_ptr(h + j * TB, r, half);
                    if (r >= s) {
                        volatile tt_l1_ptr uint32_t* src = row_ptr(x, r - s, half);
                        for (uint32_t w = 0; w < 8; ++w) {
                            dst[w] = src[w];
                        }
                    } else {
                        // u[r - s] with r - s in [-3, -1] -> carried row 3 + (r - s)
                        const uint32_t k = 3 + r - s;
                        for (uint32_t w = 0; w < 8; ++w) {
                            dst[w] = carry[k][half * 8 + w];
                        }
                    }
                }
            }
        }
        // this pass's history (h_0: row i = u[i - 3]) for the next pass
        noc_async_write(h, s_hist.get_noc_addr(c), TB);
        noc_async_write_barrier();
#else
        // shift: the register's new rows 1..3 are the old rows 2, 3 and this step's input
        noc_async_write(h + TB, s_st1.get_noc_addr(c), TB);
        noc_async_write(h + 2 * TB, s_st2.get_noc_addr(c), TB);
        noc_async_write(x, s_st3.get_noc_addr(c), TB);
        noc_async_write_barrier();  // the slots are recycled once the compute pops them
#endif
        cb_push_back(cb_h, 3);
        cb_push_back(cb_x, 1);
        cb_push_back(cb_t, 4);
    }
}
