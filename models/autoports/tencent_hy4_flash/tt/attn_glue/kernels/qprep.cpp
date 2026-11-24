// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Decode attention prep (attn_glue/__init__.py qprep): head h's real rows s < R of [q_lat | q_rope | q_sink] (TILE,
// L1) gathered as row-major rows of the sparse_sdpa query [1, Hp, 32, 608] (DRAM, page h * 32 + s). Row s of a tile:
// face (s / 16) * 2 (+1 for columns 16..31), 32 bytes at (s % 16) * 32. The sink columns are the same on every row.
// runtime args: [q_lat, q_rope, q_sink, out, h, R]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

#ifndef BATCH
#define BATCH 0  // (HY4_GLUE_BATCH) all rows staged at once: one read and one write barrier per call
#endif

void kernel_main() {
    const uint32_t ql_addr = get_arg_val<uint32_t>(0);
    const uint32_t qr_addr = get_arg_val<uint32_t>(1);
    const uint32_t qs_addr = get_arg_val<uint32_t>(2);
    const uint32_t out_addr = get_arg_val<uint32_t>(3);
    const uint32_t h = get_arg_val<uint32_t>(4);
    const uint32_t R = get_arg_val<uint32_t>(5);
    constexpr uint32_t LT = get_compile_time_arg_val(0);  // q_lat tiles per head (16)
    constexpr uint32_t RT = get_compile_time_arg_val(1);  // q_rope tiles per head (2)
    constexpr uint32_t ROW = get_compile_time_arg_val(2);  // out row bytes (608 * 2)
    constexpr uint32_t S = get_compile_time_arg_val(3);    // rows per head of out (32)
    constexpr uint32_t cb_row = get_compile_time_arg_val(4);
    constexpr auto ql_args = TensorAccessorArgs<5>();
    constexpr auto qr_args = TensorAccessorArgs<ql_args.next_compile_time_args_offset()>();
    constexpr auto qs_args = TensorAccessorArgs<qr_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<qs_args.next_compile_time_args_offset()>();
    constexpr uint32_t BF = 2048;
    const auto s_ql = TensorAccessor(ql_args, ql_addr, BF);
    const auto s_qr = TensorAccessor(qr_args, qr_addr, BF);
    const auto s_qs = TensorAccessor(qs_args, qs_addr, BF);
    const auto s_o = TensorAccessor(o_args, out_addr, ROW);
    const uint32_t l1 = get_write_ptr(cb_row);
    // the sink columns (row 0 of the sink tile: 64-byte aligned in DRAM)
    {
        const uint64_t src = s_qs.get_noc_addr(h);
        noc_async_read(src, l1 + (LT + RT) * 64, 32);
        noc_async_read(src + 512, l1 + (LT + RT) * 64 + 32, 32);
    }
#if BATCH
    // every row's reads in flight at once (row s staged at l1 + s * ROW), one read barrier, the R row writes, one write
    // barrier (was a read + write round trip per row)
    for (uint32_t s = 0; s < R; ++s) {
        const uint32_t rl = l1 + s * ROW;
        if (s > 0) {  // the sink columns (row 0's were read above)
            const uint64_t src = s_qs.get_noc_addr(h);
            noc_async_read(src, rl + (LT + RT) * 64, 32);
            noc_async_read(src + 512, rl + (LT + RT) * 64 + 32, 32);
        }
        const uint32_t off = (s / 16) * 1024 + (s % 16) * 32;
        for (uint32_t t = 0; t < LT; ++t) {
            const uint64_t src = s_ql.get_noc_addr(h * LT + t) + off;
            noc_async_read(src, rl + t * 64, 32);
            noc_async_read(src + 512, rl + t * 64 + 32, 32);
        }
        for (uint32_t t = 0; t < RT; ++t) {
            const uint64_t src = s_qr.get_noc_addr(h * RT + t) + off;
            noc_async_read(src, rl + (LT + t) * 64, 32);
            noc_async_read(src + 512, rl + (LT + t) * 64 + 32, 32);
        }
    }
    noc_async_read_barrier();
    for (uint32_t s = 0; s < R; ++s) {
        noc_async_write(l1 + s * ROW, s_o.get_noc_addr(h * S + s), ROW);
    }
    noc_async_write_barrier();
#else
    for (uint32_t s = 0; s < R; ++s) {
        const uint32_t off = (s / 16) * 1024 + (s % 16) * 32;
        for (uint32_t t = 0; t < LT; ++t) {
            const uint64_t src = s_ql.get_noc_addr(h * LT + t) + off;
            noc_async_read(src, l1 + t * 64, 32);
            noc_async_read(src + 512, l1 + t * 64 + 32, 32);
        }
        for (uint32_t t = 0; t < RT; ++t) {
            const uint64_t src = s_qr.get_noc_addr(h * RT + t) + off;
            noc_async_read(src, l1 + (LT + t) * 64, 32);
            noc_async_read(src + 512, l1 + (LT + t) * 64 + 32, 32);
        }
        noc_async_read_barrier();
        noc_async_write(l1, s_o.get_noc_addr(h * S + s), ROW);
        noc_async_write_barrier();
    }
#endif
}
