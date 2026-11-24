// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Indexer pooling (exact), reader (RISCV_1). MODE 0 (pool): per unit u = (k, ct) (row tile k < NT, column tile
// ct < 4): for slot j = 0..3 the v / g tiles of vg and of the bias (pages (j NT + k) 8 + ct and + 4) -> cb 0 (vg: v_j,
// g_j interleaved) and cb 1 (bias, same order). MODE 1 (post): the normed tile (page k 4 + ct) -> cb 0, the key-norm
// weight row tile ct -> cb 1. ct: [MODE, NT, acc(a), acc(b)]; rt: [a addr, b addr, u0, nu]
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t MODE = get_compile_time_arg_val(0);
    constexpr uint32_t NT = get_compile_time_arg_val(1);
    constexpr auto a_args = TensorAccessorArgs<2>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr uint32_t F32 = 4096;
    const uint32_t a_addr = get_arg_val<uint32_t>(0), b_addr = get_arg_val<uint32_t>(1);
    const uint32_t u0 = get_arg_val<uint32_t>(2), nu = get_arg_val<uint32_t>(3);
    const auto sa = TensorAccessor(a_args, a_addr, F32);
    const auto sb = TensorAccessor(b_args, b_addr, F32);
    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t k = u / 4, ct = u % 4;
        if constexpr (MODE == 0) {
            cb_reserve_back(0, 8);
            cb_reserve_back(1, 8);
            uint32_t wa = get_write_ptr(0), wb = get_write_ptr(1);
            for (uint32_t j = 0; j < 4; ++j) {
                const uint32_t p = (j * NT + k) * 8 + ct;
                noc_async_read(sa.get_noc_addr(p), wa, F32);
                noc_async_read(sa.get_noc_addr(p + 4), wa + F32, F32);
                noc_async_read(sb.get_noc_addr(p), wb, F32);
                noc_async_read(sb.get_noc_addr(p + 4), wb + F32, F32);
                wa += 2 * F32;
                wb += 2 * F32;
            }
            noc_async_read_barrier();
            cb_push_back(0, 8);
            cb_push_back(1, 8);
        } else {
            cb_reserve_back(0, 1);
            cb_reserve_back(1, 1);
            noc_async_read(sa.get_noc_addr(k * 4 + ct), get_write_ptr(0), F32);
            noc_async_read(sb.get_noc_addr(ct), get_write_ptr(1), F32);
            noc_async_read_barrier();
            cb_push_back(0, 1);
            cb_push_back(1, 1);
        }
    }
}
