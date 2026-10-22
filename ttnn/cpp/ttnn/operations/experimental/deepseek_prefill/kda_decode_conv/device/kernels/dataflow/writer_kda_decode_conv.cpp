// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// qkv tile c -> out page c; shifted history tiles -> h0 / h1 / h2 page c (in place: the reader read page c of this
// core's columns before the compute kernel produced them).
void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t h0_addr = get_arg_val<uint32_t>(1);
    const uint32_t h1_addr = get_arg_val<uint32_t>(2);
    const uint32_t h2_addr = get_arg_val<uint32_t>(3);
    const uint32_t start = get_arg_val<uint32_t>(4);
    const uint32_t n = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_qkv = get_compile_time_arg_val(0);
    constexpr uint32_t cb_hout = get_compile_time_arg_val(1);
    constexpr auto a_out = TensorAccessorArgs<2>();
    constexpr auto a_h0 = TensorAccessorArgs<a_out.next_compile_time_args_offset()>();
    constexpr auto a_h1 = TensorAccessorArgs<a_h0.next_compile_time_args_offset()>();
    constexpr auto a_h2 = TensorAccessorArgs<a_h1.next_compile_time_args_offset()>();

    const auto s_out = TensorAccessor(a_out, out_addr);
    const auto s_h0 = TensorAccessor(a_h0, h0_addr);
    const auto s_h1 = TensorAccessor(a_h1, h1_addr);
    const auto s_h2 = TensorAccessor(a_h2, h2_addr);

    const uint32_t fp_page = get_local_cb_interface(cb_qkv).fifo_page_size;
    const uint32_t bf_page = get_local_cb_interface(cb_hout).fifo_page_size;

    Noc noc;
    CircularBuffer cq(cb_qkv), ch(cb_hout);
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t c = start + i;
        cq.wait_front(1);
        noc.async_write(cq, s_out, fp_page, {}, {.page_id = c});
        ch.wait_front(3);
        noc.async_write(ch, s_h0, bf_page, {.offset_bytes = 0}, {.page_id = c});
        noc.async_write(ch, s_h1, bf_page, {.offset_bytes = bf_page}, {.page_id = c});
        noc.async_write(ch, s_h2, bf_page, {.offset_bytes = 2 * bf_page}, {.page_id = c});
        noc.async_write_barrier();
        cq.pop_front(1);
        ch.pop_front(3);
    }
}
