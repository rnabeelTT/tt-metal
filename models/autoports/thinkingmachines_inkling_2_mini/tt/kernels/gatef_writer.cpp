// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Prefill message-gate tail (tt/gatef.py), writer: this core's output tiles (fp32), row tile by row tile, its columns.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t o_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t c0 = get_arg_val<uint32_t>(0);
    const uint32_t nc = get_arg_val<uint32_t>(1);
    const uint32_t r0 = get_arg_val<uint32_t>(2);
    const uint32_t nr = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t F32 = 4096;
    constexpr auto o_args = TensorAccessorArgs<2>();
    const auto ao = TensorAccessor(o_args, o_addr, F32);
    for (uint32_t r = r0; r < r0 + nr; ++r) {
        for (uint32_t j = 0; j < nc; ++j) {
            cb_wait_front(cb_out, 1);
            noc_async_write(get_read_ptr(cb_out), ao.get_noc_addr(r * Ht + c0 + j), F32);
            noc_async_writes_flushed();
            cb_pop_front(cb_out, 1);
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
