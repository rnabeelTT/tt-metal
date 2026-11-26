// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Inkling LM head, activation in + logits out (BRISC; tt/head.py StreamHead): rows 0 .. rows-1 of the kt BF16
// activation tiles into cb_x (rows <= 16: faces 0 / 1, the other rows are don't-care; else whole tiles), then each FP32 output column tile the
// compute packs to logits page out_col0 + i (columns past n_tiles: bank padding, not written; HEAD_LIVE: only its live
// rows, the others are don't-care like the activation's).
// HEAD_TOPK: also each live row's candidate list of this core's columns (lhlist.hpp) as the tiles stream, written at
// the end as page SLOT * 32 + row of `lists`.
//
// runtime args: [x_addr, out_addr, out_col0, ncols(, lists_addr, slot)]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "lhkey.hpp"
#include "lhlist.hpp"


void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_col0 = get_arg_val<uint32_t>(2);
    const uint32_t ncols = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t kt = get_compile_time_arg_val(2);
    constexpr uint32_t rows = get_compile_time_arg_val(3);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(4);
    constexpr auto x_args = TensorAccessorArgs<5>();
    constexpr auto o_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    static_assert(rows <= 32, "one tile row");
    constexpr uint32_t row_bytes = rows * 32;
    constexpr uint32_t OB = 4096;  // an FP32 tile

    if (ncols == 0) {
        return;
    }
#if HEAD_TOPK
    constexpr uint32_t cb_l = get_compile_time_arg_val(o_args.next_compile_time_args_offset());
    constexpr auto l_args = TensorAccessorArgs<o_args.next_compile_time_args_offset() + 1>();
    // cb_l: the rows' buffers, the compaction scratch, the rows' list pages (off the small kernel stack)
    const uint32_t lbase = get_write_ptr(cb_l);
    auto row = [&](uint32_t r) { return reinterpret_cast<LhRow*>(lbase + r * LH_ROW_WORDS * 4); };
    int32_t* scratch = reinterpret_cast<int32_t*>(lbase + rows * LH_ROW_WORDS * 4);
    const uint32_t pages = lbase + rows * LH_ROW_WORDS * 4 + LH_CAP * 4;
    for (uint32_t r = 0; r < rows; ++r) {
        lh_row_init(row(r));
    }
#endif
    cb_reserve_back(cb_x, kt);
    const uint32_t xl1 = get_write_ptr(cb_x);
    const auto s_x = TensorAccessor(x_args, x_addr, 2048);
    for (uint32_t k = 0; k < kt; ++k) {
        const uint64_t src = s_x.get_noc_addr(k);
        if constexpr (rows > 16) {  // (all four faces)
            noc_async_read(src, xl1 + k * 2048, 2048);
        } else {
            noc_async_read(src, xl1 + k * 2048, row_bytes);
            noc_async_read(src + 512, xl1 + k * 2048 + 512, row_bytes);
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, kt);

    const auto s_o = TensorAccessor(o_args, out_addr, OB);
#if HEAD_STAGE
    static_assert(rows <= 16, "staged: faces 0 / 1");
    constexpr uint32_t cb_s = 3;
    const uint32_t stage = get_write_ptr(cb_s);
    uint32_t ns = 0, sp0 = out_col0;  // (the staged tiles are consecutive pages from sp0)
    auto flush_stage = [&]() {
        noc_async_write_barrier();  // (the local copies landed)
        for (uint32_t k = 0; k < ns; ++k) {
            const uint32_t src = stage + k * 2 * rows * 64;
            const uint64_t dst = s_o.get_noc_addr(sp0 + k);
            noc_async_write(src, dst, rows * 64);
            noc_async_write(src + rows * 64, dst + 1024, rows * 64);
        }
        sp0 += ns;
        ns = 0;
        noc_async_writes_flushed();  // (the staging buffer read out before it is refilled)
    };
#endif
    for (uint32_t i = 0; i < ncols; ++i) {
        cb_wait_front(cb_out, 1);
        if (out_col0 + i < n_tiles) {
#if HEAD_STAGE
            // the live rows copied (on the NOC) into a staging buffer and written HEAD_STAGE tiles at a time: the
            // weight stream's banks see a few write bursts instead of a write per tile
            {
                const uint32_t a = get_read_ptr(cb_out);
                const uint32_t d = stage + ns * 2 * rows * 64;
                noc_async_write(a, get_noc_addr(d), rows * 64);  // (a local copy on the NOC)
                noc_async_write(a + 1024, get_noc_addr(d + rows * 64), rows * 64);
                if (++ns == HEAD_STAGE) {
                    flush_stage();
                }
            }
#elif HEAD_LIVE
            // the live rows only (faces 0 / 1 rows 0 .. rows-1; past 16 rows faces 0 / 1 whole, faces 2 / 3 the rest)
            const uint32_t src = get_read_ptr(cb_out);
            const uint64_t dst = s_o.get_noc_addr(out_col0 + i);
            if constexpr (rows > 16) {
                noc_async_write(src, dst, 2048);
                noc_async_write(src + 2048, dst + 2048, (rows - 16) * 64);
                noc_async_write(src + 3072, dst + 3072, (rows - 16) * 64);
            } else {
                noc_async_write(src, dst, rows * 64);
                noc_async_write(src + 1024, dst + 1024, rows * 64);
            }
#else
            noc_async_write(get_read_ptr(cb_out), s_o.get_noc_addr(out_col0 + i), OB);
#endif
        }
#if HEAD_TOPK
        invalidate_l1_cache();
        const uint32_t base = get_read_ptr(cb_out);
        const uint32_t col0 = (out_col0 + i) * 32;
        for (uint32_t r = 0; r < rows; ++r) {
            LhRow* lr = row(r);
            for (uint32_t f = 0; f < 2; ++f) {
                const uint32_t* p = reinterpret_cast<const uint32_t*>(base + ((r >> 4) * 2 + f) * 1024 + (r & 15) * 64);
                const int32_t t = lr->thr();
                // prefilter: the OR of (t / 2 - 1 - v / 2) has its sign set if some value may pass (halved: no wrap)
                uint32_t any = 0;
                if (t >= 0) {  // (the raw bits order the passing values)
                    const int32_t h = (t >> 1) - 1;
#pragma GCC unroll 16
                    for (uint32_t j = 0; j < 16; ++j) {
                        any |= static_cast<uint32_t>(h - (static_cast<int32_t>(p[j]) >> 1));
                    }
                } else {  // (on halved unsigned orders)
                    const uint32_t h = ((static_cast<uint32_t>(t) ^ 0x80000000u) >> 1) - 1;
#pragma GCC unroll 16
                    for (uint32_t j = 0; j < 16; ++j) {
                        any |= h - ((static_cast<uint32_t>(lh_sord(p[j])) ^ 0x80000000u) >> 1);
                    }
                }
                if ((any & 0x80000000u) == 0) {
                    continue;
                }
                const uint32_t c0 = col0 + f * 16;
                for (uint32_t j = 0; j < 16; ++j) {
                    const int32_t v = lh_sord(p[j]);
                    if (v > lr->thr()) {
                        lh_row_add(lr, v, c0 + j, scratch);
                    }
                }
            }
        }
#endif
        noc_async_writes_flushed();
        cb_pop_front(cb_out, 1);
    }
#if HEAD_TOPK
    const auto s_l = TensorAccessor(l_args, get_arg_val<uint32_t>(4), LH_PAGE);
    const uint32_t slot = get_arg_val<uint32_t>(5);
    for (uint32_t r = 0; r < rows; ++r) {
        uint32_t* pg = reinterpret_cast<uint32_t*>(pages + r * LH_PAGE);
        lh_row_finish(row(r), scratch, pg);
        noc_async_write(reinterpret_cast<uint32_t>(pg), s_l.get_noc_addr(slot * 32 + r), LH_PAGE);
    }
#endif
#if HEAD_STAGE
    if (ns) {
        flush_stage();
    }
#endif
    noc_async_write_barrier();
}
