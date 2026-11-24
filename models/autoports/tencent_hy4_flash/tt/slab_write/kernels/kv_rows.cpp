// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Tiled parts -> row-major cache rows (data movement only; HY4_KVROW_FUSED): row k (k = c, c + NCORES, ... < NR on
// core c) of the parts' single tile row, the parts' columns side by side ([a | b], A + B tiles, bf16), then ZB zero
// bytes, is written to destination row offs[k] (offs: one uint32 page read at run time: trace-safe). Replaces concat +
// to_layout + slab_write (the same bytes). Row r of a tile: faces 0 / 1 (r < 16) or 2 / 3, 32 bytes each.
//
// runtime args: [a, b, dst, offs, c]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t dst_addr = get_arg_val<uint32_t>(2);
    const uint32_t offs_addr = get_arg_val<uint32_t>(3);
    const uint32_t c = get_arg_val<uint32_t>(4);

    constexpr uint32_t A = get_compile_time_arg_val(0);      // tiles of part a
    constexpr uint32_t B = get_compile_time_arg_val(1);      // tiles of part b
    constexpr uint32_t ZB = get_compile_time_arg_val(2);     // zero bytes after the parts
    constexpr uint32_t ROWB = get_compile_time_arg_val(3);   // destination row bytes (= 64 (A + B) + ZB)
    constexpr uint32_t OFFB = get_compile_time_arg_val(4);   // bytes of the offsets page
    constexpr uint32_t cb_row = get_compile_time_arg_val(5);
    constexpr uint32_t cb_offs = get_compile_time_arg_val(6);
    constexpr uint32_t NR = get_compile_time_arg_val(7);
    constexpr uint32_t NCORES = get_compile_time_arg_val(8);
    constexpr auto a_args = TensorAccessorArgs<9>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto dst_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr auto offs_args = TensorAccessorArgs<dst_args.next_compile_time_args_offset()>();
    static_assert(ROWB == 64 * (A + B) + ZB);

    const auto s_a = TensorAccessor(a_args, a_addr, 2048);
    const auto s_b = TensorAccessor(b_args, b_addr, 2048);
    const auto s_dst = TensorAccessor(dst_args, dst_addr, ROWB);
    const auto s_offs = TensorAccessor(offs_args, offs_addr, OFFB);

    const uint32_t ol1 = get_write_ptr(cb_offs);
    noc_async_read(s_offs.get_noc_addr(0), ol1, OFFB);
    const uint32_t rl1 = get_write_ptr(cb_row);
    volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rl1 + 64 * (A + B));
    for (uint32_t i = 0; i < ZB / 4; ++i) {
        z[i] = 0;
    }
    noc_async_read_barrier();
    for (uint32_t k = c; k < NR; k += NCORES) {
        const uint32_t off = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ol1)[k];
        const uint32_t f0 = (k < 16 ? 0 : 1024) + (k % 16) * 32;  // the row's bytes in face 0 (2) and face 1 (3)
        for (uint32_t t = 0; t < A; ++t) {
            const uint64_t s = s_a.get_noc_addr(t);
            noc_async_read(s + f0, rl1 + t * 64, 32);
            noc_async_read(s + f0 + 512, rl1 + t * 64 + 32, 32);
        }
        for (uint32_t t = 0; t < B; ++t) {
            const uint64_t s = s_b.get_noc_addr(t);
            noc_async_read(s + f0, rl1 + (A + t) * 64, 32);
            noc_async_read(s + f0 + 512, rl1 + (A + t) * 64 + 32, 32);
        }
        noc_async_read_barrier();
        noc_async_write(rl1, s_dst.get_noc_addr(off), ROWB);
        noc_async_write_barrier();
    }
}
