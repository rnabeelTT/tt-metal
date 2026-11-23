// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Grouped expert matmul (tt/gmm.py), writer: the nb output tiles of each (group, row tile) -> C tiles
// (g mt + mi) Nt + b nb + j.
//
// runtime args: b, g0, g1      common args: c

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t Nt = get_compile_time_arg_val(1);
    constexpr uint32_t nb = get_compile_time_arg_val(2);
    constexpr uint32_t mt = get_compile_time_arg_val(3);
    constexpr auto c_args = TensorAccessorArgs<4>();
    const uint32_t b = get_arg_val<uint32_t>(0), g0 = get_arg_val<uint32_t>(1), g1 = get_arg_val<uint32_t>(2);
    const auto ac = TensorAccessor(c_args, get_common_arg_val<uint32_t>(0), 2048);
    for (uint32_t g = g0; g < g1; ++g) {
        for (uint32_t mi = 0; mi < mt; ++mi) {
            cb_wait_front(cb_out, nb);
            const uint32_t l = get_read_ptr(cb_out);
            for (uint32_t j = 0; j < nb; ++j) {
                noc_async_write(l + j * 2048, ac.get_noc_addr((g * mt + mi) * Nt + b * nb + j), 2048);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, nb);
        }
    }
}
