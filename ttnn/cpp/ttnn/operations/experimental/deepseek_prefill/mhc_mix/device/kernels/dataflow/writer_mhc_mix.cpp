// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Unit u = r * C + c -> output [1, C, T, 64]: tiles (c, r, 0) projection partial, (c, r, 1) Gram partial.
void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_units = get_arg_val<uint32_t>(1);
    const uint32_t start = get_arg_val<uint32_t>(2);
    constexpr uint32_t Tt = get_compile_time_arg_val(0);
    constexpr uint32_t C = get_compile_time_arg_val(1);
    constexpr auto out_args = TensorAccessorArgs<2>();
    constexpr uint32_t CB_OUT = 16;
    const uint32_t page = get_local_cb_interface(CB_OUT).fifo_page_size;
    const auto s_out = TensorAccessor(out_args, out_addr);
    Noc noc;
    CircularBuffer co(CB_OUT);
    for (uint32_t w = 0; w < num_units; ++w) {
        const uint32_t u = start + w;
        const uint32_t r = u / C;
        const uint32_t c = u - r * C;
        co.wait_front(2);
        for (uint32_t j = 0; j < 2; ++j) {
            noc.async_write(co, s_out, page, {.offset_bytes = j * page}, {.page_id = (c * Tt + r) * 2 + j});
        }
        noc.async_write_barrier();
        co.pop_front(2);
    }
}
