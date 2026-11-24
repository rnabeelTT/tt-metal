// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Merged MoE un-dispatch (tt/undisp.py), writer: each item's Wt_c summed tiles -> output tiles (rt, cs * Wt_c + i).
//
// runtime args: item_start, item_count      common args: out

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t Wt_c = get_compile_time_arg_val(1);
    constexpr uint32_t col_splits = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr auto out_args = TensorAccessorArgs<4>();
    const uint32_t item_start = get_arg_val<uint32_t>(0);
    const uint32_t item_count = get_arg_val<uint32_t>(1);
    const auto s_out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(0), 2048);
    for (uint32_t it = 0; it < item_count; ++it) {
        const uint32_t item = item_start + it;
        const uint32_t tile0 = (item / col_splits) * Wt + (item % col_splits) * Wt_c;
        cb_wait_front(cb_out, Wt_c);
        const uint32_t l1 = get_read_ptr(cb_out);
        for (uint32_t i = 0; i < Wt_c; ++i) {
            noc_async_write(l1 + i * 2048, s_out.get_noc_addr(tile0 + i), 2048);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, Wt_c);
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
