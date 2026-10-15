// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Unit u = r * C + c: the Kc x tiles of row tile r in K chunk c, and the chunk's Kc projection tiles (fn_t is one tile
// column wide: page = K tile index). The projection tiles are re-read only when the chunk changes.
void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t fn_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_units = get_arg_val<uint32_t>(2);
    const uint32_t start = get_arg_val<uint32_t>(3);
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Kc = get_compile_time_arg_val(1);
    constexpr uint32_t C = get_compile_time_arg_val(2);
    constexpr auto x_args = TensorAccessorArgs<3>();
    constexpr auto fn_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr uint32_t CB_X = 0, CB_FN = 1;
    const uint32_t page = get_local_cb_interface(CB_X).fifo_page_size;
    const auto s_x = TensorAccessor(x_args, x_addr);
    const auto s_fn = TensorAccessor(fn_args, fn_addr);
    Noc noc;
    CircularBuffer cx(CB_X), cfn(CB_FN);
    for (uint32_t w = 0; w < num_units; ++w) {
        const uint32_t u = start + w;
        const uint32_t r = u / C;
        const uint32_t c = u - r * C;
        cx.reserve_back(Kc);
        cfn.reserve_back(Kc);
        for (uint32_t k = 0; k < Kc; ++k) {
            noc.async_read(s_x, cx, page, {.page_id = r * Kt + c * Kc + k}, {.offset_bytes = k * page});
            noc.async_read(s_fn, cfn, page, {.page_id = c * Kc + k}, {.offset_bytes = k * page});
        }
        noc.async_read_barrier();
        cx.push_back(Kc);
        cfn.push_back(Kc);
    }
}
