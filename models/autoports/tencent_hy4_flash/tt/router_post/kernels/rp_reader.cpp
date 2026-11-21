// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Router post, reader: column tiles [t0, t1) of a, b and the row-replicated bias (FP32 tile pages).
// runtime args: [a, b, bias, t0, t1]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t c_addr = get_arg_val<uint32_t>(2);
    const uint32_t t0 = get_arg_val<uint32_t>(3);
    const uint32_t t1 = get_arg_val<uint32_t>(4);
    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr uint32_t F32 = 4096, cb_a = 0, cb_b = 1, cb_c = 2;
    const auto s_a = TensorAccessor(a_args, a_addr, F32);
    const auto s_b = TensorAccessor(b_args, b_addr, F32);
    const auto s_c = TensorAccessor(c_args, c_addr, F32);
    for (uint32_t t = t0; t < t1; ++t) {
        cb_reserve_back(cb_a, 1);
        cb_reserve_back(cb_b, 1);
        cb_reserve_back(cb_c, 1);
        noc_async_read(s_a.get_noc_addr(t), get_write_ptr(cb_a), F32);
        noc_async_read(s_b.get_noc_addr(t), get_write_ptr(cb_b), F32);
        noc_async_read(s_c.get_noc_addr(t), get_write_ptr(cb_c), F32);
        noc_async_read_barrier();
        cb_push_back(cb_a, 1);
        cb_push_back(cb_b, 1);
        cb_push_back(cb_c, 1);
    }
}
