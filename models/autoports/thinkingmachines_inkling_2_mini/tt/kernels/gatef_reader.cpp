// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Prefill message-gate tail with its forget / gain matmul (tt/gatef.py), reader. This core: hidden columns
// c0 .. c0 + nc - 1, row tiles r0 .. r0 + nr - 1. First the [forget | gain] weight tiles of its columns (bf16,
// W [R, 2H]: per column j the R forget tiles (k, c0 + j) then the R gain tiles (k, Ht + c0 + j)), kept; then per row
// tile the R tiles of g (bf16 [T, R]) and per column the h / m tiles (fp32).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t g_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t h_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t m_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t c0 = get_arg_val<uint32_t>(0);
    const uint32_t nc = get_arg_val<uint32_t>(1);
    const uint32_t r0 = get_arg_val<uint32_t>(2);
    const uint32_t nr = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_g = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_hm = get_compile_time_arg_val(2);
    constexpr uint32_t Ht = get_compile_time_arg_val(3);
    constexpr uint32_t R = get_compile_time_arg_val(4);
    constexpr uint32_t F32 = 4096, BF = 2048;
    constexpr auto g_args = TensorAccessorArgs<5>();
    constexpr auto w_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto h_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto m_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    const auto ag = TensorAccessor(g_args, g_addr, BF);
    const auto aw = TensorAccessor(w_args, w_addr, BF);
    const auto ah = TensorAccessor(h_args, h_addr, F32);
    const auto am = TensorAccessor(m_args, m_addr, F32);

    cb_reserve_back(cb_w, 2 * R * nc);
    {
        uint32_t p = get_write_ptr(cb_w);
        for (uint32_t j = 0; j < nc; ++j) {
            for (uint32_t half = 0; half < 2; ++half) {
                for (uint32_t k = 0; k < R; ++k) {
                    noc_async_read(aw.get_noc_addr(k * 2 * Ht + half * Ht + c0 + j), p, BF);
                    p += BF;
                }
            }
        }
        noc_async_read_barrier();
    }
    cb_push_back(cb_w, 2 * R * nc);
    for (uint32_t r = r0; r < r0 + nr; ++r) {
        cb_reserve_back(cb_g, R);
        const uint32_t pg = get_write_ptr(cb_g);
        for (uint32_t k = 0; k < R; ++k) {
            noc_async_read(ag.get_noc_addr(r * R + k), pg + k * BF, BF);
        }
        noc_async_read_barrier();
        cb_push_back(cb_g, R);
        for (uint32_t j = 0; j < nc; ++j) {
            cb_reserve_back(cb_hm, 2);
            const uint32_t p = get_write_ptr(cb_hm);
            noc_async_read(ah.get_noc_addr(r * Ht + c0 + j), p, F32);
            noc_async_read(am.get_noc_addr(r * Ht + c0 + j), p + F32, F32);
            noc_async_read_barrier();
            cb_push_back(cb_hm, 2);
        }
    }
}
