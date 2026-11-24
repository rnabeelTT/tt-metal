// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Block attention residual mix (tt/mixw.py), writer: per row tile, column t, query q: the output tile (this core's H range) to out_q (fp32).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t nq = get_compile_time_arg_val(1);
    constexpr uint32_t F32 = 4096;
    constexpr auto o_args = TensorAccessorArgs<2>();
    const uint32_t t0 = get_arg_val<uint32_t>(0);
    const uint32_t nt = get_arg_val<uint32_t>(1);
    const uint32_t r0 = get_arg_val<uint32_t>(2);
    const uint32_t nr = get_arg_val<uint32_t>(3);
    const uint32_t Ht = get_arg_val<uint32_t>(4);
    for (uint32_t r = r0; r < r0 + nr; ++r) {
        for (uint32_t t = 0; t < nt; ++t) {
            for (uint32_t q = 0; q < nq; ++q) {
                const auto ao = TensorAccessor(o_args, get_arg_val<uint32_t>(5 + q), F32);
                cb_wait_front(cb_out, 1);
                const uint64_t d = ao.get_noc_addr(r * Ht + t0 + t);
#if defined(MIX_TH) && MIX_TH < 32  // (the live rows of faces 0 / 1 of a MIX_TH x 32 CB tile)
                noc_async_write(get_read_ptr(cb_out), d, MIX_T * 64);
                noc_async_write(get_read_ptr(cb_out) + MIX_TH * 64, d + 1024, MIX_T * 64);
#else
                noc_async_write(get_read_ptr(cb_out), d, F32);
#endif
                noc_async_writes_flushed();
                cb_pop_front(cb_out, 1);
            }
        }
    }
    noc_async_write_barrier();
}
