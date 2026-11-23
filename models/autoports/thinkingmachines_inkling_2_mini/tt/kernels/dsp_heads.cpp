// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// DSpark drafter head split / merge (tt/drafter.py _heads): a tile copy between a row-major activation
// [1, 1, RT*32, CT*32] and a head-major one [1, nh, RT*32, DTH*32] (bf16 TILE, DRAM interleaved):
//   head-major tile (h, r, d)  <->  row-major tile (r, c0 + h*DTH + d)
// mode 0 (split): row-major -> head-major; mode 1 (merge): head-major -> row-major. Each core copies head-major linear
// tiles t0 .. t0+nt-1 of one (src, dst) pair.
//
// runtime args: [src_addr, dst_addr, c0, RT, CT, t0, nt, mode]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t TB = 2048;

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t c0 = get_arg_val<uint32_t>(2);
    const uint32_t RT = get_arg_val<uint32_t>(3);
    const uint32_t CT = get_arg_val<uint32_t>(4);
    const uint32_t t0 = get_arg_val<uint32_t>(5);
    const uint32_t nt = get_arg_val<uint32_t>(6);
    const uint32_t mode = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t DTH = get_compile_time_arg_val(1);
    constexpr uint32_t NBUF = get_compile_time_arg_val(2);
    constexpr auto s_args = TensorAccessorArgs<3>();
    constexpr auto d_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();

    const auto s_s = TensorAccessor(s_args, src_addr, TB);
    const auto s_d = TensorAccessor(d_args, dst_addr, TB);
    const uint32_t l1 = get_write_ptr(cb);
    for (uint32_t b = 0; b < nt; b += NBUF) {
        const uint32_t n = nt - b < NBUF ? nt - b : NBUF;
        for (uint32_t j = 0; j < n; ++j) {
            const uint32_t t = t0 + b + j;  // head-major linear tile
            const uint32_t d = t % DTH, hr = t / DTH, r = hr % RT, h = hr / RT;
            const uint32_t rm = r * CT + c0 + h * DTH + d;
            noc_async_read(s_s.get_noc_addr(mode ? t : rm), l1 + j * TB, TB);
        }
        noc_async_read_barrier();
        for (uint32_t j = 0; j < n; ++j) {
            const uint32_t t = t0 + b + j;
            const uint32_t d = t % DTH, hr = t / DTH, r = hr % RT, h = hr / RT;
            const uint32_t rm = r * CT + c0 + h * DTH + d;
            noc_async_write(l1 + j * TB, s_d.get_noc_addr(mode ? rm : t), TB);
        }
        noc_async_write_barrier();
    }
}
