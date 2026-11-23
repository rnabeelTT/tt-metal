// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Block attention residual mix (tt/mixw.py), reader: every logit tile (n sources x nq queries; each query's logit in
// every column of its tile) and this core's H tiles of the n stacked sources (fp32), per row tile.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t s_addr = get_arg_val<uint32_t>(0);
    const uint32_t lg_addr = get_arg_val<uint32_t>(1);
    const uint32_t t0 = get_arg_val<uint32_t>(2);
    const uint32_t nt = get_arg_val<uint32_t>(3);
    const uint32_t r0 = get_arg_val<uint32_t>(4);  // this core's row tiles r0 .. r0 + nr - 1
    const uint32_t nr = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_lg = get_compile_time_arg_val(0);
    constexpr uint32_t cb_s = get_compile_time_arg_val(1);
    constexpr uint32_t n = get_compile_time_arg_val(2);
    constexpr uint32_t nq = get_compile_time_arg_val(3);
    constexpr uint32_t Ht = get_compile_time_arg_val(4);
    constexpr uint32_t RT = get_compile_time_arg_val(5);  // row tiles (T / 32)
    constexpr uint32_t F32 = 4096;
    constexpr auto s_args = TensorAccessorArgs<6>();
    constexpr auto lg_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    const auto as = TensorAccessor(s_args, s_addr, F32);
    const auto alg = TensorAccessor(lg_args, lg_addr, F32);

    for (uint32_t r = r0; r < r0 + nr; ++r) {
        // logits of row tile r: [1, n, T, nq * 32] tiles (i * RT + r) * nq + q
        cb_reserve_back(cb_lg, n * nq);
        const uint32_t l = get_write_ptr(cb_lg);
        for (uint32_t i = 0; i < n; ++i) {
            for (uint32_t q = 0; q < nq; ++q) {
                noc_async_read(alg.get_noc_addr((i * RT + r) * nq + q), l + (i * nq + q) * F32, F32);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_lg, n * nq);
        // this core's H tiles of row tile r: [1, n, T, H] tiles (i * RT + r) * Ht + t, the n sources of column t
        for (uint32_t t = 0; t < nt; ++t) {
            cb_reserve_back(cb_s, n);
            const uint32_t sp = get_write_ptr(cb_s);
            for (uint32_t i = 0; i < n; ++i) {
                noc_async_read(as.get_noc_addr((i * RT + r) * Ht + t0 + t), sp + i * F32, F32);
            }
            noc_async_read_barrier();
            cb_push_back(cb_s, n);
        }
    }
}
