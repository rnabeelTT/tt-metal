// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Packed mix logits -> the replicated layout (tt/lgx.py): out tile (s, q) = column q % 32 of in tile (s, q / 32) in
// all 32 columns, rows 0 .. T-1 only (fp32 [1, n, T <= 32, W] interleaved). The values are copied, not computed.
//
// runtime args: in addr, out addr, first item, item count (item = s Nq + q); compile time: cb, in tiles per row (Wp),
// out tiles per row (Nq), then the two accessors

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t WP = get_compile_time_arg_val(1);
    constexpr uint32_t NQ = get_compile_time_arg_val(2);
    constexpr uint32_t T = LGX_T;
    constexpr uint32_t T0 = T < 16 ? T : 16, T1 = T > 16 ? T - 16 : 0;  // (rows in faces 0 / 1, faces 2 / 3)
    constexpr auto i_args = TensorAccessorArgs<3>();
    constexpr auto o_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    const auto ai = TensorAccessor(i_args, get_arg_val<uint32_t>(0), 4096);
    const auto ao = TensorAccessor(o_args, get_arg_val<uint32_t>(1), 4096);
    const uint32_t i0 = get_arg_val<uint32_t>(2), ni = get_arg_val<uint32_t>(3);
    const uint32_t in = get_write_ptr(cb), ob = in + 4096;  // (in tile, then two out tiles)
    uint32_t cur = 0xFFFFFFFFu;
    for (uint32_t k = 0; k < ni; ++k) {
        const uint32_t it = i0 + k, s = it / NQ, q = it % NQ, src = s * WP + q / 32;
        if (src != cur) {
            noc_async_writes_flushed();  // (the in tile is not read by writes, but keep the out buffers' reuse simple)
            const uint64_t a = ai.get_noc_addr(src);
            for (uint32_t f = 0; f < 2; ++f) {
                noc_async_read(a + f * 1024, in + f * 1024, T0 * 64);
                if constexpr (T1) {
                    noc_async_read(a + 2048 + f * 1024, in + 2048 + f * 1024, T1 * 64);
                }
            }
            noc_async_read_barrier();
            cur = src;
        }
        const uint32_t o = ob + (k & 1) * 4096;
        if (k >= 2) {
            noc_async_writes_flushed();  // (buffer k & 1 was sent two items ago)
        }
        const uint32_t j = q % 32, fo = (j < 16 ? 0 : 1024) + (j % 16) * 4;
        for (uint32_t r = 0; r < T; ++r) {
            const uint32_t rb = (r < 16 ? 0 : 2048) + (r % 16) * 64;
            const uint32_t v = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in + rb + fo);
            volatile tt_l1_ptr uint32_t* w0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(o + rb);
            volatile tt_l1_ptr uint32_t* w1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(o + rb + 1024);
            for (uint32_t c = 0; c < 16; ++c) {
                w0[c] = v;
                w1[c] = v;
            }
        }
        const uint64_t d = ao.get_noc_addr(s * NQ + q);
        for (uint32_t f = 0; f < 2; ++f) {
            noc_async_write(o + f * 1024, d + f * 1024, T0 * 64);
            if constexpr (T1) {
                noc_async_write(o + 2048 + f * 1024, d + 2048 + f * 1024, T1 * 64);
            }
        }
    }
    noc_async_write_barrier();
}
