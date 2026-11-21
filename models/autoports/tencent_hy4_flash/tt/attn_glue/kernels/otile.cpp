// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Decode attention output (attn_glue/__init__.py otile): the sparse_sdpa output [1, Hp, 32, V] row-major (DRAM, page
// h * 32 + s) -> [1, Hd, 32, V] TILE: this core's NT column tiles [t0, t0 + NT) of head h, rows s < R placed, every
// other row zero.
// runtime args: [o, out, h, t0, R]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t o_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t h = get_arg_val<uint32_t>(2);
    const uint32_t t0 = get_arg_val<uint32_t>(3);
    const uint32_t R = get_arg_val<uint32_t>(4);
    constexpr uint32_t NT = get_compile_time_arg_val(0);   // column tiles per core
    constexpr uint32_t VT = get_compile_time_arg_val(1);   // column tiles per head row (V / 32)
    constexpr uint32_t S = get_compile_time_arg_val(2);    // rows per head of o (32)
    constexpr uint32_t cb_t = get_compile_time_arg_val(3);  // NT tiles
    constexpr uint32_t cb_r = get_compile_time_arg_val(4);  // one row segment (NT * 64 bytes)
    constexpr auto i_args = TensorAccessorArgs<5>();
    constexpr auto o_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    constexpr uint32_t BF = 2048;
    const auto s_i = TensorAccessor(i_args, o_addr, VT * 64);
    const auto s_o = TensorAccessor(o_args, out_addr, BF);
    const uint32_t tl = get_write_ptr(cb_t);
    const uint32_t rl = get_write_ptr(cb_r);
    {
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        uint32_t dst = tl, n = NT * BF;
        while (n) {
            const uint32_t k = n < MEM_ZEROS_SIZE ? n : MEM_ZEROS_SIZE;
            noc_async_read(zeros, dst, k);
            dst += k;
            n -= k;
        }
        noc_async_read_barrier();
    }
    for (uint32_t s = 0; s < R; ++s) {
        noc_async_read(s_i.get_noc_addr(h * S + s) + t0 * 64, rl, NT * 64);
        noc_async_read_barrier();
        const uint32_t off = (s / 16) * 1024 + (s % 16) * 32;
        for (uint32_t i = 0; i < NT; ++i) {
            const uint64_t src = get_noc_addr(rl + i * 64);
            noc_async_read(src, tl + i * BF + off, 32);
            noc_async_read(src + 32, tl + i * BF + off + 512, 32);
        }
        noc_async_read_barrier();
    }
    for (uint32_t i = 0; i < NT; ++i) {
        noc_async_write(tl + i * BF, s_o.get_noc_addr(h * VT + t0 + i), BF);
    }
    noc_async_write_barrier();
}
