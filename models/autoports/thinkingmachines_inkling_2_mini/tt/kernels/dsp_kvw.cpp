// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// DSpark drafter context-cache write (tt/drafter.py): rows 0 .. R-1 of one head of a new K or V [1, NKV, R, 128] bf16
// TILE tensor go to cache rows (p0 + i) mod S of the same head ([1, NKV, S, 128] bf16 TILE; p0 read from a uint32
// [1, 1] ROW_MAJOR tensor). One (cache, head) per core; each touched source / cache tile row is read once, the cache
// rows patched and written back.
//
// runtime args: [src_addr, cache_addr, p0_addr, head, S]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t TB = 2048;

static inline void copy_row(uint32_t src_tile, uint32_t sr, uint32_t dst_tile, uint32_t dr) {
    // a tile row = 16 bf16 in face (r / 16) * 2 and 16 in the next face
    for (uint32_t f = 0; f < 2; ++f) {
        volatile tt_l1_ptr uint32_t* s =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_tile + ((sr >> 4) * 2 + f) * 512 + (sr & 15) * 32);
        volatile tt_l1_ptr uint32_t* d =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_tile + ((dr >> 4) * 2 + f) * 512 + (dr & 15) * 32);
        for (uint32_t w = 0; w < 8; ++w) {
            d[w] = s[w];
        }
    }
}

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t cache_addr = get_arg_val<uint32_t>(1);
    const uint32_t p0_addr = get_arg_val<uint32_t>(2);
    const uint32_t head = get_arg_val<uint32_t>(3);
    const uint32_t S = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb = get_compile_time_arg_val(0);  // 9 tiles: 4 source, 4 cache, 1 scratch
    constexpr uint32_t R = get_compile_time_arg_val(1);
    constexpr uint32_t DT = get_compile_time_arg_val(2);  // tiles per row (head_dim / 32)
    constexpr uint32_t RT = (R + 31) / 32;               // source tile rows per head
    constexpr auto s_args = TensorAccessorArgs<3>();
    constexpr auto c_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();

    const uint32_t base = get_write_ptr(cb);
    const uint32_t src = base, dst = base + DT * TB, scratch = base + 2 * DT * TB;
    const auto s_s = TensorAccessor(s_args, src_addr, TB);
    const auto s_c = TensorAccessor(c_args, cache_addr, TB);
    const auto s_p = TensorAccessor(p_args, p0_addr, 4);
    noc_async_read(s_p.get_noc_addr(0), scratch, 4);
    noc_async_read_barrier();
    const uint32_t p0 = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    const uint32_t rows_t = S / 32;
    uint32_t cur = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < R; ++i) {
        if ((i & 31) == 0) {
            for (uint32_t d = 0; d < DT; ++d) {
                noc_async_read(s_s.get_noc_addr((head * RT + (i >> 5)) * DT + d), src + d * TB, TB);
            }
            noc_async_read_barrier();
        }
        const uint32_t s = (p0 + i) % S;
        const uint32_t tr = s >> 5;
        if (tr != cur) {
            if (cur != 0xFFFFFFFFu) {
                for (uint32_t d = 0; d < DT; ++d) {
                    noc_async_write(dst + d * TB, s_c.get_noc_addr((head * rows_t + cur) * DT + d), TB);
                }
                noc_async_write_barrier();
            }
            for (uint32_t d = 0; d < DT; ++d) {
                noc_async_read(s_c.get_noc_addr((head * rows_t + tr) * DT + d), dst + d * TB, TB);
            }
            noc_async_read_barrier();
            cur = tr;
        }
        for (uint32_t d = 0; d < DT; ++d) {
            copy_row(src + d * TB, i & 31, dst + d * TB, s & 31);
        }
    }
    for (uint32_t d = 0; d < DT; ++d) {
        noc_async_write(dst + d * TB, s_c.get_noc_addr((head * rows_t + cur) * DT + d), TB);
    }
    noc_async_write_barrier();
}
