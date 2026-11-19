// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Multi-slab row write (data movement only): core k copies slab k (rows [k S, k S + S) of the row-major source) to
// the row-major destination at row offs[k] (offs: one uint32 page, read at run time: trace-safe). Pages are rows.
//
// runtime args: [src, dst, offs, k]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t offs_addr = get_arg_val<uint32_t>(2);
    const uint32_t k = get_arg_val<uint32_t>(3);

    constexpr uint32_t S = get_compile_time_arg_val(0);       // rows per slab
    constexpr uint32_t ROWB = get_compile_time_arg_val(1);    // bytes per row (page)
    constexpr uint32_t OFFB = get_compile_time_arg_val(2);    // bytes of the offsets page
    constexpr uint32_t cb_rows = get_compile_time_arg_val(3);
    constexpr uint32_t cb_offs = get_compile_time_arg_val(4);
    constexpr auto src_args = TensorAccessorArgs<5>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto offs_args = TensorAccessorArgs<dst_args.next_compile_time_args_offset()>();

    const auto s_src = TensorAccessor(src_args, src_addr, ROWB);
    const auto s_dst = TensorAccessor(dst_args, dst_addr, ROWB);
    const auto s_offs = TensorAccessor(offs_args, offs_addr, OFFB);

    const uint32_t ol1 = get_write_ptr(cb_offs);
    noc_async_read(s_offs.get_noc_addr(0), ol1, OFFB);
    const uint32_t rl1 = get_write_ptr(cb_rows);
    for (uint32_t i = 0; i < S; ++i) {
        noc_async_read(s_src.get_noc_addr(k * S + i), rl1 + i * ROWB, ROWB);
    }
    noc_async_read_barrier();
    const uint32_t off = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ol1)[k];
    for (uint32_t i = 0; i < S; ++i) {
        noc_async_write(rl1 + i * ROWB, s_dst.get_noc_addr(off + i), ROWB);
    }
    noc_async_write_barrier();
}
