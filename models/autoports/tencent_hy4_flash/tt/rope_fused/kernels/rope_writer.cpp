// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fused RoPE, writer (RISCV_0): y tiles [t0, t1). runtime args: [y, t0, t1]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t t0 = get_arg_val<uint32_t>(1);
    const uint32_t t1 = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_y = 16;
    constexpr auto y_args = TensorAccessorArgs<0>();
    constexpr uint32_t BF = 2048;
    const auto s_y = TensorAccessor(y_args, y_addr, BF);
    for (uint32_t t = t0; t < t1; ++t) {
        cb_wait_front(cb_y, 1);
        noc_async_write(get_read_ptr(cb_y), s_y.get_noc_addr(t), BF);
        noc_async_write_barrier();
        cb_pop_front(cb_y, 1);
    }
}
