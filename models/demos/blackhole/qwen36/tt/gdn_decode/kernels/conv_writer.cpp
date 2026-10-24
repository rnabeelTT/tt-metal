// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// GDN decode short conv, writer: the FP32 conv output tiles into conv_out ([1, 1, 32, qkv] FP32 TILE, page c).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t c0 = get_arg_val<uint32_t>(1);
    const uint32_t n = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_o = get_compile_time_arg_val(0);
    constexpr uint32_t TB = 4096;  // fp32 tile
    constexpr auto out_args = TensorAccessorArgs<1>();
    const auto s_out = TensorAccessor(out_args, out_addr, TB);
    for (uint32_t i = 0; i < n; ++i) {
        cb_wait_front(cb_o, 1);
        noc_async_write(get_read_ptr(cb_o), s_out.get_noc_addr(c0 + i), TB);
        noc_async_write_barrier();
        cb_pop_front(cb_o, 1);
    }
}
