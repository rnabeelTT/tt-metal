// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Prefill expert SwiGLU, writer: NT output tiles per group to [1, 1, Rd, Il] (tile g * NT + j).  runtime: [out, g0, g1]
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t o_addr = get_arg_val<uint32_t>(0);
    const uint32_t g0 = get_arg_val<uint32_t>(1);
    const uint32_t g1 = get_arg_val<uint32_t>(2);
    constexpr uint32_t NT = get_compile_time_arg_val(0);
    constexpr auto o_args = TensorAccessorArgs<1>();
    constexpr uint32_t PG = 2048, cb_out = 16;
    const auto s_o = TensorAccessor(o_args, o_addr, PG);
    for (uint32_t g = g0; g < g1; ++g) {
        for (uint32_t j = 0; j < NT; ++j) {
            cb_wait_front(cb_out, 1);
            noc_async_write(get_read_ptr(cb_out), s_o.get_noc_addr(g * NT + j), PG);
            noc_async_writes_flushed();
            cb_pop_front(cb_out, 1);
        }
    }
    noc_async_write_barrier();
}
