// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Inkling short conv over a prefill chunk, writer (pf_reader.cpp's work split): output tile (k, c) = page k * cols + c.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    constexpr uint32_t cb_o = get_compile_time_arg_val(0);
    constexpr uint32_t cols = get_compile_time_arg_val(1);
    constexpr uint32_t Rt = get_compile_time_arg_val(2);
    constexpr uint32_t G = get_compile_time_arg_val(3);
    constexpr uint32_t GY = get_compile_time_arg_val(4);
    constexpr uint32_t TB = get_tile_size(cb_o);  // (bf16, or fp32 with the packed kernel's residual add)
    constexpr auto out_args = TensorAccessorArgs<5>();

    const uint32_t item = (uint32_t)get_absolute_logical_x() * GY + (uint32_t)get_absolute_logical_y();
    if (item >= cols * G) {
        return;
    }
    const uint32_t c = item % cols;
    constexpr uint32_t per = (Rt + G - 1) / G;
    const uint32_t k0 = (item / cols) * per;
    const uint32_t k1 = (k0 + per < Rt) ? k0 + per : Rt;
    const auto s_out = TensorAccessor(out_args, out_addr, TB);
    for (uint32_t k = k0; k < k1; ++k) {
        cb_wait_front(cb_o, 1);
        noc_async_write(get_read_ptr(cb_o), s_out.get_noc_addr(k * cols + c), TB);
        noc_async_write_barrier();
        cb_pop_front(cb_o, 1);
    }
}
