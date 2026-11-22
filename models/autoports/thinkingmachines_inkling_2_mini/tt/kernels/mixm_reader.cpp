// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Block attention residual mix (tt/mixw.py MixKernel.multi), reader for one row tile (T <= 32): the n sources and
// their logits come from separate tensors (no stacked copy): source i's H tiles from S_i, its nq logit tiles from
// tiles off_i .. off_i + nq - 1 of LG_i (a block's logits against every later query, or the current source's own).
// Same CB layout as mixw_reader.cpp (logit tile i * nq + q; source tiles ordered [t][i]).
//
// runtime args: t0, nt, then per source i: s_addr, lg_addr, lg_off

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t t0 = get_arg_val<uint32_t>(0);
    const uint32_t nt = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_lg = get_compile_time_arg_val(0);
    constexpr uint32_t cb_s = get_compile_time_arg_val(1);
    constexpr uint32_t n = get_compile_time_arg_val(2);
    constexpr uint32_t nq = get_compile_time_arg_val(3);
    constexpr uint32_t F32 = 4096;
    constexpr auto s_args = TensorAccessorArgs<4>();
    constexpr auto lg_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();

    cb_reserve_back(cb_lg, n * nq);
    const uint32_t l = get_write_ptr(cb_lg);
    for (uint32_t i = 0; i < n; ++i) {
        const auto alg = TensorAccessor(lg_args, get_arg_val<uint32_t>(2 + 3 * i + 1), F32);
        const uint32_t off = get_arg_val<uint32_t>(2 + 3 * i + 2);
        for (uint32_t q = 0; q < nq; ++q) {
            noc_async_read(alg.get_noc_addr(off + q), l + (i * nq + q) * F32, F32);
        }
    }
    cb_reserve_back(cb_s, n * nt);
    const uint32_t sp = get_write_ptr(cb_s);
    for (uint32_t i = 0; i < n; ++i) {
        const auto as = TensorAccessor(s_args, get_arg_val<uint32_t>(2 + 3 * i), F32);
        for (uint32_t t = 0; t < nt; ++t) {
            noc_async_read(as.get_noc_addr(t0 + t), sp + (t * n + i) * F32, F32);
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_lg, n * nq);
    cb_push_back(cb_s, n * nt);
}
