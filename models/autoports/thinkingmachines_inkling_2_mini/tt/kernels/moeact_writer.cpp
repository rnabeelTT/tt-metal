// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Prefill MoE activation (tt/moeact.py), writer: tile (r, c) of the [Rd, W] output, row tiles [r0, r1).
//
// runtime args: r0, r1      common args: out

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr auto o_args = TensorAccessorArgs<2>();
    const uint32_t r0 = get_arg_val<uint32_t>(0), r1 = get_arg_val<uint32_t>(1);
    const auto ao = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0), 2048);
    for (uint32_t r = r0; r < r1; ++r) {
        for (uint32_t c = 0; c < Wt; ++c) {
            cb_wait_front(cb_out, 1);
            noc_async_write(get_read_ptr(cb_out), ao.get_noc_addr(r * Wt + c), 2048);
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
    }
}
