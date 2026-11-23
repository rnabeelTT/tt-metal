// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// DSpark drafter control (tt/drafter.py; one core): from p0 (first commit row's position) and n (committed rows), both
// uint32 [1, 1] ROW_MAJOR, the draft position p = p0 + n and
//   idx_c  uint32 [1, RC]: p0 + i               (commit rows' RoPE rows)
//   idx_n  uint32 [1, 1]:  max(n, 1) - 1        (the committed row whose taps feed f_rnn)
//   idx_b  uint32 [1, B]:  p + i                (block rows' RoPE rows; element 0 = the block's first cache row)
//   cpos   int32 [NU]:     p + B - 1            (the SDPA's last key: the block rows sit at cache rows p .. p+B-1)
//   wp0    w1[anchor] as row 0 of [1, 1, 1, 256] bf16 TILE (the first Markov step's input)
//
// runtime args: [p0_addr, n_addr, ic_addr, in_addr, ib_addr, cp_addr, anchor_addr, w1_addr, wp0_addr]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "dsp_w1row.hpp"

void kernel_main() {
    const uint32_t p0_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_addr = get_arg_val<uint32_t>(1);
    const uint32_t ic_addr = get_arg_val<uint32_t>(2);
    const uint32_t in_addr = get_arg_val<uint32_t>(3);
    const uint32_t ib_addr = get_arg_val<uint32_t>(4);
    const uint32_t cp_addr = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t RC = get_compile_time_arg_val(1);
    constexpr uint32_t B = get_compile_time_arg_val(2);
    constexpr uint32_t NU = get_compile_time_arg_val(3);
    constexpr auto a1 = TensorAccessorArgs<4>();
    constexpr auto a2 = TensorAccessorArgs<a1.next_compile_time_args_offset()>();
    constexpr auto a3 = TensorAccessorArgs<a2.next_compile_time_args_offset()>();
    constexpr auto a4 = TensorAccessorArgs<a3.next_compile_time_args_offset()>();
    constexpr auto a5 = TensorAccessorArgs<a4.next_compile_time_args_offset()>();
    constexpr auto a6 = TensorAccessorArgs<a5.next_compile_time_args_offset()>();
    constexpr auto a7 = TensorAccessorArgs<a6.next_compile_time_args_offset()>();
    constexpr auto a8 = TensorAccessorArgs<a7.next_compile_time_args_offset()>();
    constexpr auto a9 = TensorAccessorArgs<a8.next_compile_time_args_offset()>();

    const uint32_t l1 = get_write_ptr(cb);  // [0, 64): p0, [64, 128): n, [256, 384): index rows, [512, 640): cpos
    noc_async_read(TensorAccessor(a1, p0_addr, 4).get_noc_addr(0), l1, 4);
    noc_async_read(TensorAccessor(a2, n_addr, 4).get_noc_addr(0), l1 + 64, 4);
    noc_async_read(TensorAccessor(a7, get_arg_val<uint32_t>(6), 4).get_noc_addr(0), l1 + 128, 4);
    noc_async_read_barrier();
    const uint32_t p0 = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    const uint32_t n = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1 + 64);
    const uint32_t p = p0 + n;
    volatile tt_l1_ptr uint32_t* ix = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1 + 256);
    volatile tt_l1_ptr uint32_t* cp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1 + 512);

    for (uint32_t i = 0; i < RC; ++i) {
        ix[i] = p0 + i;
    }
    noc_async_write(l1 + 256, TensorAccessor(a3, ic_addr, RC * 4).get_noc_addr(0), RC * 4);
    noc_async_write_barrier();
    ix[0] = n > 0 ? n - 1 : 0;
    noc_async_write(l1 + 256, TensorAccessor(a4, in_addr, 4).get_noc_addr(0), 4);
    noc_async_write_barrier();
    for (uint32_t i = 0; i < B; ++i) {
        ix[i] = p + i;
    }
    noc_async_write(l1 + 256, TensorAccessor(a5, ib_addr, B * 4).get_noc_addr(0), B * 4);
    for (uint32_t i = 0; i < NU; ++i) {
        cp[i] = p + B - 1;
    }
    noc_async_write(l1 + 512, TensorAccessor(a6, cp_addr, NU * 4).get_noc_addr(0), NU * 4);
    noc_async_write_barrier();
    w1_row(TensorAccessor(a8, get_arg_val<uint32_t>(7), 512), TensorAccessor(a9, get_arg_val<uint32_t>(8), 2048),
           *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1 + 128), l1 + 1024);
}
