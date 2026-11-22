// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Message gate tail (tt/gatet.py), reader: per H tile t of this core, the pre-tanh forget / gain tiles (fg tiles t and
// Ht + t: fg = [forget | gain] over 2H columns, one tile row), h tile t, m tile t (all fp32).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t fg_addr = get_arg_val<uint32_t>(0);
    const uint32_t h_addr = get_arg_val<uint32_t>(1);
    const uint32_t m_addr = get_arg_val<uint32_t>(2);
    const uint32_t t0 = get_arg_val<uint32_t>(3);
    const uint32_t nt = get_arg_val<uint32_t>(4);
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t F32 = 4096;
    constexpr auto fg_args = TensorAccessorArgs<2>();
    constexpr auto h_args = TensorAccessorArgs<fg_args.next_compile_time_args_offset()>();
    constexpr auto m_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    const auto afg = TensorAccessor(fg_args, fg_addr, F32);
    const auto ah = TensorAccessor(h_args, h_addr, F32);
    const auto am = TensorAccessor(m_args, m_addr, F32);
    for (uint32_t t = t0; t < t0 + nt; ++t) {
        cb_reserve_back(cb_in, 4);
        const uint32_t p = get_write_ptr(cb_in);
        noc_async_read(afg.get_noc_addr(t), p, F32);
        noc_async_read(afg.get_noc_addr(Ht + t), p + F32, F32);
        noc_async_read(ah.get_noc_addr(t), p + 2 * F32, F32);
        noc_async_read(am.get_noc_addr(t), p + 3 * F32, F32);
        noc_async_read_barrier();
        cb_push_back(cb_in, 4);
    }
}
