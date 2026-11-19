// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Multi-slab write (data movement only). Slab k (k = c, c + NCORES, ... < NS on core c) moves to the destination at
// row offs[k] (offs: one uint32 page read at run time: trace-safe).
//   rows mode (TILEM = 0): row-major tensors, pages are rows: src rows [k S, k S + S) -> dst rows [off, off + S).
//   tile mode (TILEM = 1): tiled tensors, one 32-row tile row of WT tiles: src tile row k -> dst tile row off / 32.
//
// runtime args: [src, dst, offs, c]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t offs_addr = get_arg_val<uint32_t>(2);
    const uint32_t c = get_arg_val<uint32_t>(3);

    constexpr uint32_t S = get_compile_time_arg_val(0);       // pages per slab (rows mode: rows; tile mode: WT)
    constexpr uint32_t PAGEB = get_compile_time_arg_val(1);   // bytes per page
    constexpr uint32_t OFFB = get_compile_time_arg_val(2);    // bytes of the offsets page
    constexpr uint32_t cb_rows = get_compile_time_arg_val(3);
    constexpr uint32_t cb_offs = get_compile_time_arg_val(4);
    constexpr uint32_t NS = get_compile_time_arg_val(5);
    constexpr uint32_t NCORES = get_compile_time_arg_val(6);
    constexpr uint32_t TILEM = get_compile_time_arg_val(7);
    constexpr auto src_args = TensorAccessorArgs<8>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto offs_args = TensorAccessorArgs<dst_args.next_compile_time_args_offset()>();

    const auto s_src = TensorAccessor(src_args, src_addr, PAGEB);
    const auto s_dst = TensorAccessor(dst_args, dst_addr, PAGEB);
    const auto s_offs = TensorAccessor(offs_args, offs_addr, OFFB);

    const uint32_t ol1 = get_write_ptr(cb_offs);
    noc_async_read(s_offs.get_noc_addr(0), ol1, OFFB);
    noc_async_read_barrier();
    const uint32_t rl1 = get_write_ptr(cb_rows);
    for (uint32_t k = c; k < NS; k += NCORES) {
        const uint32_t off = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ol1)[k];
        const uint32_t d0 = TILEM ? (off / 32) * S : off;
        for (uint32_t i = 0; i < S; ++i) {
            noc_async_read(s_src.get_noc_addr(k * S + i), rl1 + i * PAGEB, PAGEB);
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < S; ++i) {
            noc_async_write(rl1 + i * PAGEB, s_dst.get_noc_addr(d0 + i), PAGEB);
        }
        noc_async_write_barrier();
    }
}
