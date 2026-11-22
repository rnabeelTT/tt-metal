// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Block attention residual mix (tt/mixw.py), writer: query q's output tiles (this core's H range) to out_q (fp32).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t nq = get_compile_time_arg_val(1);
    constexpr uint32_t F32 = 4096;
    constexpr auto o_args = TensorAccessorArgs<2>();
    const uint32_t t0 = get_arg_val<uint32_t>(0);
    const uint32_t nt = get_arg_val<uint32_t>(1);
    for (uint32_t q = 0; q < nq; ++q) {
        const auto ao = TensorAccessor(o_args, get_arg_val<uint32_t>(2 + q), F32);
        for (uint32_t t = 0; t < nt; ++t) {
            cb_wait_front(cb_out, 1);
            noc_async_write(get_read_ptr(cb_out), ao.get_noc_addr(t0 + t), F32);
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
    }
}
