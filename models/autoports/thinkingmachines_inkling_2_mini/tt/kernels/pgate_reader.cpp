// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Prefill attention output gate (tt/agate.py PrefillGate), reader: this core's tiles [t0, t1) of o [1, 1, T, Wt * 32]
// bf16 and the matching gate tiles of pq [1, 1, T, PQt * 32] fp32 (from tile column G0).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_o = get_compile_time_arg_val(0);
    constexpr uint32_t cb_g = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(2);
    constexpr uint32_t PQt = get_compile_time_arg_val(3);
    constexpr uint32_t G0 = get_compile_time_arg_val(4);
    constexpr auto o_args = TensorAccessorArgs<5>();
    constexpr auto g_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    const uint32_t t0 = get_arg_val<uint32_t>(0), t1 = get_arg_val<uint32_t>(1);
    const auto ao = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0), 2048);
    const auto ag = TensorAccessor(g_args, get_common_arg_val<uint32_t>(1), 4096);
    for (uint32_t t = t0; t < t1; ++t) {
        const uint32_t r = t / Wt, c = t % Wt;
        cb_reserve_back(cb_o, 1);
        cb_reserve_back(cb_g, 1);
        noc_async_read(ao.get_noc_addr(t), get_write_ptr(cb_o), 2048);
        noc_async_read(ag.get_noc_addr(r * PQt + G0 + c), get_write_ptr(cb_g), 4096);
        noc_async_read_barrier();
        cb_push_back(cb_o, 1);
        cb_push_back(cb_g, 1);
    }
}
