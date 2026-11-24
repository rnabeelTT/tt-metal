// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Prefill expert SwiGLU, reader: per 32-row group g in [g0, g1): its 2 * NT [up | gate] tiles of the grouped gate|up
// matmul output and its routing-weight tile (column 0 of [Rd, 1] bf16).   runtime args: [fused, w, g0, g1]
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t f_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t g0 = get_arg_val<uint32_t>(2);
    const uint32_t g1 = get_arg_val<uint32_t>(3);
    constexpr uint32_t NT = get_compile_time_arg_val(0);  // tiles per half (Il / 32)
    constexpr auto f_args = TensorAccessorArgs<1>();
    constexpr auto w_args = TensorAccessorArgs<f_args.next_compile_time_args_offset()>();
    constexpr uint32_t PG = 2048, cb_in = 0, cb_w = 1;
    const auto s_f = TensorAccessor(f_args, f_addr, PG);
    const auto s_w = TensorAccessor(w_args, w_addr, PG);
    for (uint32_t g = g0; g < g1; ++g) {
        cb_reserve_back(cb_w, 1);
        noc_async_read(s_w.get_noc_addr(g), get_write_ptr(cb_w), PG);
        cb_reserve_back(cb_in, 2 * NT);
        uint32_t l1 = get_write_ptr(cb_in);
        for (uint32_t c = 0; c < 2 * NT; ++c) {
            noc_async_read(s_f.get_noc_addr(g * 2 * NT + c), l1 + c * PG, PG);
        }
        noc_async_read_barrier();
        cb_push_back(cb_w, 1);
        cb_push_back(cb_in, 2 * NT);
    }
}
