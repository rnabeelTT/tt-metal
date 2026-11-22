// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Attention output gate (tt/agate.py), writer: rows b < B of core j's output tile (bf16) -> tile j of the output.
//
// common args: out

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t CX = get_compile_time_arg_val(1);
    constexpr uint32_t B = AG_ROWS;
    constexpr auto a = TensorAccessorArgs<2>();
    const uint32_t j = (uint32_t)get_absolute_logical_y() * CX + (uint32_t)get_absolute_logical_x();
    const auto ao = TensorAccessor(a, get_common_arg_val<uint32_t>(0), 2048);
    cb_wait_front(cb_out, 1);
    const uint32_t s = get_read_ptr(cb_out);
    const uint64_t d = ao.get_noc_addr(j);
    noc_async_write(s, d, B * 32);
    noc_async_write(s + 512, d + 512, B * 32);
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
}
