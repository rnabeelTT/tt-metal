// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Full-width QK-norm statistics, writer: this core's row-sum tile (FP32) to page `slot` of the [1, 1, 32, P * 32]
// statistics tensor.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t slot = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr auto out_args = TensorAccessorArgs<1>();
    constexpr uint32_t F32_TILE = 4096;
    const auto s_out = TensorAccessor(out_args, out_addr, F32_TILE);
    cb_wait_front(cb_out, 1);
    noc_async_write(get_read_ptr(cb_out), s_out.get_noc_addr(slot), F32_TILE);
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
}
