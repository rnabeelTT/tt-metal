// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fused RoPE, reader (RISCV_1): the rotate-half tile P (once), then per x tile t in [t0, t1): x tile t (bf16), and the
// cos / sin tiles (fp32) of its row tile and column (tile ((t / 2) % TR) * 2 + t % 2).
//
// runtime args: [x, cos, sin, p, t0, t1]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t c_addr = get_arg_val<uint32_t>(1);
    const uint32_t s_addr = get_arg_val<uint32_t>(2);
    const uint32_t p_addr = get_arg_val<uint32_t>(3);
    const uint32_t t0 = get_arg_val<uint32_t>(4);
    const uint32_t t1 = get_arg_val<uint32_t>(5);

    constexpr uint32_t TR = get_compile_time_arg_val(0);  // row tiles of x / the tables
    constexpr uint32_t WT = get_compile_time_arg_val(1);  // x tiles per tile row (2; tail mode: W / 32, last 2 roped)
    constexpr uint32_t cb_x = 0, cb_p = 1, cb_c = 2, cb_s = 3;
    constexpr auto x_args = TensorAccessorArgs<2>();
    constexpr auto c_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr uint32_t BF = 2048, F32 = 4096;

    const auto s_p = TensorAccessor(p_args, p_addr, BF);
    cb_reserve_back(cb_p, 1);
    noc_async_read(s_p.get_noc_addr(0), get_write_ptr(cb_p), BF);
    noc_async_read_barrier();
    cb_push_back(cb_p, 1);

    const auto s_x = TensorAccessor(x_args, x_addr, BF);
    const auto s_c = TensorAccessor(c_args, c_addr, F32);
    const auto s_s = TensorAccessor(s_args, s_addr, F32);
    for (uint32_t t = t0; t < t1; ++t) {
        const uint32_t ct = ((t >> 1) % TR) * 2 + (t & 1);
        cb_reserve_back(cb_x, 1);
        cb_reserve_back(cb_c, 1);
        cb_reserve_back(cb_s, 1);
        noc_async_read(s_x.get_noc_addr((t >> 1) * WT + WT - 2 + (t & 1)), get_write_ptr(cb_x), BF);
        noc_async_read(s_c.get_noc_addr(ct), get_write_ptr(cb_c), F32);
        noc_async_read(s_s.get_noc_addr(ct), get_write_ptr(cb_s), F32);
        noc_async_read_barrier();
        cb_push_back(cb_x, 1);
        cb_push_back(cb_c, 1);
        cb_push_back(cb_s, 1);
    }
}
