// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// DSpark drafter tap mixing, data movement (tt/drafter.py _mix): per assigned tile (r, c) of the [R, H] tap rows, the 4
// taps' bf16 tiles (taps [1, 4, RT*32, CT*32] TILE) into cb_in, then the NM mixed FP32 tiles from cb_out to the NM
// outputs ([1, 1, RT*32, CT*32] FP32 TILE each).
//
// runtime args: [taps_addr, t0, nt, out_addr_0 .. out_addr_{NM-1}]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t taps_addr = get_arg_val<uint32_t>(0);
    const uint32_t t0 = get_arg_val<uint32_t>(1);
    const uint32_t nt = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t NM = get_compile_time_arg_val(2);
    constexpr uint32_t TPT = get_compile_time_arg_val(3);  // tiles per tap ([R, H] tile count)
    constexpr auto t_args = TensorAccessorArgs<4>();
    constexpr auto o_args = TensorAccessorArgs<t_args.next_compile_time_args_offset()>();

    const auto s_t = TensorAccessor(t_args, taps_addr, 2048);
    for (uint32_t i = 0; i < nt; ++i) {
        const uint32_t t = t0 + i;
        cb_reserve_back(cb_in, 4);
        const uint32_t l1 = get_write_ptr(cb_in);
        for (uint32_t j = 0; j < 4; ++j) {
            noc_async_read(s_t.get_noc_addr(j * TPT + t), l1 + j * 2048, 2048);
        }
        noc_async_read_barrier();
        cb_push_back(cb_in, 4);
        cb_wait_front(cb_out, NM);
        const uint32_t o = get_read_ptr(cb_out);
        for (uint32_t m = 0; m < NM; ++m) {
            const auto s_o = TensorAccessor(o_args, get_arg_val<uint32_t>(3 + m), 4096);
            noc_async_write(o + m * 4096, s_o.get_noc_addr(t), 4096);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, NM);
    }
}
