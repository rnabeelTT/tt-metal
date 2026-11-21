// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Router post, writer: key tile t -> output page t, s tile t -> page n + t (the [keys | s] row, one tile row).
// runtime args: [y, t0, t1]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t t0 = get_arg_val<uint32_t>(1);
    const uint32_t t1 = get_arg_val<uint32_t>(2);
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr auto y_args = TensorAccessorArgs<1>();
    constexpr uint32_t F32 = 4096, cb_k = 16, cb_s = 17;
    const auto s_y = TensorAccessor(y_args, y_addr, F32);
    for (uint32_t t = t0; t < t1; ++t) {
        cb_wait_front(cb_k, 1);
        cb_wait_front(cb_s, 1);
        noc_async_write(get_read_ptr(cb_k), s_y.get_noc_addr(t), F32);
        noc_async_write(get_read_ptr(cb_s), s_y.get_noc_addr(n + t), F32);
        noc_async_write_barrier();
        cb_pop_front(cb_k, 1);
        cb_pop_front(cb_s, 1);
    }
}
