// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Message gate tail (tt/gatet.py), writer: this core's output tiles (fp32).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t o_addr = get_arg_val<uint32_t>(0);
    const uint32_t t0 = get_arg_val<uint32_t>(1);
    const uint32_t nt = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t F32 = 4096;
    constexpr auto o_args = TensorAccessorArgs<1>();
    const auto ao = TensorAccessor(o_args, o_addr, F32);
    for (uint32_t t = t0; t < t0 + nt; ++t) {
        cb_wait_front(cb_out, 1);
        noc_async_write(get_read_ptr(cb_out), ao.get_noc_addr(t), F32);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
