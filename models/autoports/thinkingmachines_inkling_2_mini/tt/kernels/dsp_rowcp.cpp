// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// DSpark drafter row copy (tt/drafter.py _rowcp): row r of tiles t0 .. t0+nt-1 of a TILE tensor -> row 0 of the same
// tiles of another (the other rows of the destination are left as they are). r = row, plus element 0 of a uint32 index
// tensor when idx_addr != 0. One job per core.
//
// runtime args: [src_addr, dst_addr, idx_addr, row, t0, nt, wt (tiles per tile row)]
// compile args: cb (scratch: 128 + 256 nt bytes), src / dst / idx TensorAccessorArgs

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t TB = 2048;

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t idx_addr = get_arg_val<uint32_t>(2);
    uint32_t r = get_arg_val<uint32_t>(3);
    const uint32_t t0 = get_arg_val<uint32_t>(4);
    const uint32_t nt = get_arg_val<uint32_t>(5);
    const uint32_t wt = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr auto s_args = TensorAccessorArgs<1>();
    constexpr auto d_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr auto i_args = TensorAccessorArgs<d_args.next_compile_time_args_offset()>();

    const uint32_t l1 = (get_write_ptr(cb) + 63) & ~63u;
    if (idx_addr != 0) {
        noc_async_read(TensorAccessor(i_args, idx_addr, 4).get_noc_addr(0), l1, 4);
        noc_async_read_barrier();
        r += *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    }
    const auto s_s = TensorAccessor(s_args, src_addr, TB);
    const auto s_d = TensorAccessor(d_args, dst_addr, TB);
    // row r: 16 values in face (r / 16) * 2 at (r % 16) * 32 bytes, 16 in the next face; read as 64-byte aligned chunks
    const uint32_t tr = (r >> 5) * wt;
    r &= 31;
    const uint32_t off = ((r >> 4) * 2) * 512 + (r & 15) * 32;
    const uint32_t a = off & ~63u, h = off & 63u;
    const uint32_t rd = l1 + 64, st = rd + 128 * nt;
    for (uint32_t i = 0; i < nt; ++i) {
        const uint64_t s = s_s.get_noc_addr(tr + t0 + i);
        noc_async_read(s + a, rd + i * 128, 64);
        noc_async_read(s + a + 512, rd + i * 128 + 64, 64);
    }
    noc_async_read_barrier();
    for (uint32_t i = 0; i < nt; ++i) {
        const uint64_t d = s_d.get_noc_addr(t0 + i);
        for (uint32_t f = 0; f < 2; ++f) {
            volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rd + i * 128 + f * 64 + h);
            const uint32_t sl = st + (2 * i + f) * 64;  // (64-aligned, like the destination row: faces 0 / 1, row 0)
            volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sl);
            for (uint32_t w = 0; w < 8; ++w) {
                dst[w] = src[w];
            }
            noc_async_write(sl, d + f * 512, 32);
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
