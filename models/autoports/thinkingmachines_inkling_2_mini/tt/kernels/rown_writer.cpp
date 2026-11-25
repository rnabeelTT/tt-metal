// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Prefill row norm (tt/rown.py), writer: each output tile of row tile r, column t to every destination k (same layout
// and dtype): tile r * stride_k + off_k + t (a destination may be a column range of a wider tensor, e.g. one half of
// a gate's [rms(h) | rms(m)] input).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t r0 = get_arg_val<uint32_t>(0);
    const uint32_t nr = get_arg_val<uint32_t>(1);
    // common args: nd, the destinations (addr, stride, off) at 1 + 3 k, then the logits' address (ROWN_LG)
    const uint32_t nd = get_common_arg_val<uint32_t>(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t ob = get_compile_time_arg_val(2);
    constexpr auto o_args = TensorAccessorArgs<3>();
    for (uint32_t r = r0; r < r0 + nr; ++r) {
        for (uint32_t t = 0; t < Wt; t += 4) {  // (the compute pushes 4-tile blocks)
            cb_wait_front(cb_out, 4);
            const uint32_t src = get_read_ptr(cb_out);
            for (uint32_t k = 0; k < nd; ++k) {
                const auto ao = TensorAccessor(o_args, get_common_arg_val<uint32_t>(1 + 3 * k), ob);
                const uint32_t i = r * get_common_arg_val<uint32_t>(2 + 3 * k) + get_common_arg_val<uint32_t>(3 + 3 * k) + t;
                for (uint32_t j = 0; j < 4; ++j) {
                    noc_async_write(src + j * ob, ao.get_noc_addr(i + j), ob);
                }
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_out, 4);
        }
#ifdef ROWN_LG
        {  // the row tile's NQ logit tiles (fp32) -> lg [T, NQ * 32]
            constexpr uint32_t cb_lg = get_compile_time_arg_val(o_args.next_compile_time_args_offset());
            constexpr uint32_t NQ = get_compile_time_arg_val(o_args.next_compile_time_args_offset() + 1);
            constexpr auto l_args = TensorAccessorArgs<o_args.next_compile_time_args_offset() + 2>();
            const auto al = TensorAccessor(l_args, get_common_arg_val<uint32_t>(1 + 3 * nd), 4096);
            cb_wait_front(cb_lg, NQ);
            for (uint32_t q = 0; q < NQ; ++q) {
                noc_async_write(get_read_ptr(cb_lg) + q * 4096, al.get_noc_addr(r * NQ + q), 4096);
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_lg, NQ);
        }
#endif
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
