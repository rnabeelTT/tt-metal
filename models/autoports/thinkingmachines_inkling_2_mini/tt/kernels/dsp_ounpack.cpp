// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// DSpark drafter attention output unpack (tt/drafter.py _ounpack): the SDPA output [1, NKV, G R, 128] bf16 TILE (item g
// = KV head, rows j R + i = query head g G + j, block row i) -> [1, 1, R, NH 128] bf16 TILE (the o_proj input; rows
// past R not written: row-local consumers only). One core per source tile (g, d): read once, its G heads' rows to G output tiles. Replaces the reshape
// and the head merge.
//
// runtime args: [src, dst, g, d]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t TB = 2048;

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t g = get_arg_val<uint32_t>(2);
    const uint32_t d = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb = get_compile_time_arg_val(0);  // 1 + G tiles
    constexpr uint32_t R = get_compile_time_arg_val(1);
    constexpr uint32_t G = get_compile_time_arg_val(2);
    constexpr uint32_t DT = get_compile_time_arg_val(3);
    constexpr auto s_args = TensorAccessorArgs<4>();
    constexpr auto d_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    const auto as = TensorAccessor(s_args, src_addr, TB);
    const auto ad = TensorAccessor(d_args, dst_addr, TB);
    const uint32_t src = get_write_ptr(cb), out = src + TB;
    noc_async_read(as.get_noc_addr(g * DT + d), src, TB);
    noc_async_read_barrier();
    for (uint32_t j = 0; j < G; ++j) {
        for (uint32_t i = 0; i < R; ++i) {
            const uint32_t sr = j * R + i;
            for (uint32_t f = 0; f < 2; ++f) {
                volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    src + ((sr >> 4) * 2 + f) * 512 + (sr & 15) * 32);
                volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    out + j * TB + ((i >> 4) * 2 + f) * 512 + (i & 15) * 32);
                for (uint32_t w = 0; w < 8; ++w) {
                    o[w] = s[w];
                }
            }
        }
        noc_async_write(out + j * TB, ad.get_noc_addr((g * G + j) * DT + d), TB);
    }
    noc_async_write_barrier();
}
