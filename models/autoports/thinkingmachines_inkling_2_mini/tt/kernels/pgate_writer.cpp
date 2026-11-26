// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Prefill attention output gate (tt/agate.py PrefillGate), writer: this core's output tiles [t0, t1).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr auto a = TensorAccessorArgs<1>();
    const uint32_t t0 = get_arg_val<uint32_t>(0), t1 = get_arg_val<uint32_t>(1);
    const auto ao = TensorAccessor(a, get_common_arg_val<uint32_t>(0), 2048);
    for (uint32_t t = t0; t < t1; ++t) {
        cb_wait_front(cb_out, 1);
        noc_async_write(get_read_ptr(cb_out), ao.get_noc_addr(t), 2048);
        noc_async_writes_flushed();
        cb_pop_front(cb_out, 1);
    }
    noc_async_write_barrier();
}
