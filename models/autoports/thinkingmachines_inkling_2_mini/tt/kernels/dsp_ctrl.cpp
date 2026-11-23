// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// DSpark drafter control (tt/drafter.py; one core): from p0 (first commit row's position) and n (committed rows), both
// uint32 [1, 1] ROW_MAJOR, the draft position p = p0 + n and
//   idx_c  uint32 [1, RC]: p0 + i               (commit rows' RoPE rows)
//   idx_n  uint32 [1, 1]:  max(n, 1) - 1        (the committed row whose taps feed f_rnn)
//   idx_b  uint32 [1, 32]: p + i (i < B), else p (block rows' RoPE rows)
//   gmask  FP32 TILE [1, 1, 1, SG + 32] (row 0): global layer, cache col c valid iff c < p; block col j iff j < B
//   smask  FP32 TILE [1, 1, 1, SW + 32]: ring of SW rows, col c holds position p-1 - ((p-1-c) mod SW): valid iff
//          (p-1-c) mod SW < W and <= p-1; block cols as above.  (0 = valid, NEG bits otherwise)
//
// runtime args: [p0_addr, n_addr, ic_addr, in_addr, ib_addr, gm_addr, sm_addr]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

template <typename A>
static inline void mask_row(const A& s, uint32_t l1, uint32_t ncols_ctx, int32_t p, uint32_t ring, uint32_t W,
                            uint32_t B, uint32_t NEG) {
    volatile tt_l1_ptr uint32_t* v = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    const uint32_t tiles = ncols_ctx / 32 + 1;
    for (uint32_t t = 0; t < tiles; ++t) {
        for (uint32_t j = 0; j < 32; ++j) {
            const uint32_t c = t * 32 + j;
            bool ok;
            if (c >= ncols_ctx) {
                ok = (c - ncols_ctx) < B;
            } else if (!ring) {
                ok = (int32_t)c < p;
            } else {
                int32_t d = (p - 1 - (int32_t)c) % (int32_t)ncols_ctx;
                if (d < 0) {
                    d += ncols_ctx;
                }
                ok = d < (int32_t)W && d <= p - 1;
            }
            v[(t & 1) * 32 + j] = ok ? 0u : NEG;
        }
        const uint64_t dst = s.get_noc_addr(t);
        noc_async_write(l1 + (t & 1) * 128, dst, 64);
        noc_async_write(l1 + (t & 1) * 128 + 64, dst + 1024, 64);
        noc_async_writes_flushed();
    }
}

void kernel_main() {
    const uint32_t p0_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_addr = get_arg_val<uint32_t>(1);
    const uint32_t ic_addr = get_arg_val<uint32_t>(2);
    const uint32_t in_addr = get_arg_val<uint32_t>(3);
    const uint32_t ib_addr = get_arg_val<uint32_t>(4);
    const uint32_t gm_addr = get_arg_val<uint32_t>(5);
    const uint32_t sm_addr = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t RC = get_compile_time_arg_val(1);
    constexpr uint32_t B = get_compile_time_arg_val(2);
    constexpr uint32_t SG = get_compile_time_arg_val(3);
    constexpr uint32_t SW = get_compile_time_arg_val(4);
    constexpr uint32_t W = get_compile_time_arg_val(5);
    constexpr uint32_t NEG = get_compile_time_arg_val(6);
    constexpr auto a1 = TensorAccessorArgs<7>();
    constexpr auto a2 = TensorAccessorArgs<a1.next_compile_time_args_offset()>();
    constexpr auto a3 = TensorAccessorArgs<a2.next_compile_time_args_offset()>();
    constexpr auto a4 = TensorAccessorArgs<a3.next_compile_time_args_offset()>();
    constexpr auto a5 = TensorAccessorArgs<a4.next_compile_time_args_offset()>();
    constexpr auto a6 = TensorAccessorArgs<a5.next_compile_time_args_offset()>();
    constexpr auto a7 = TensorAccessorArgs<a6.next_compile_time_args_offset()>();

    const uint32_t l1 = get_write_ptr(cb);  // [0, 64): p0, [64, 128): n, [256, 512): index rows, [512, 768): mask
    const auto s_p0 = TensorAccessor(a1, p0_addr, 4);
    const auto s_n = TensorAccessor(a2, n_addr, 4);
    noc_async_read(s_p0.get_noc_addr(0), l1, 4);
    noc_async_read(s_n.get_noc_addr(0), l1 + 64, 4);
    noc_async_read_barrier();
    const uint32_t p0 = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    const uint32_t n = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1 + 64);
    const uint32_t p = p0 + n;
    volatile tt_l1_ptr uint32_t* ix = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1 + 256);

    for (uint32_t i = 0; i < RC; ++i) {
        ix[i] = p0 + i;
    }
    noc_async_write(l1 + 256, TensorAccessor(a3, ic_addr, RC * 4).get_noc_addr(0), RC * 4);
    noc_async_write_barrier();
    ix[0] = n > 0 ? n - 1 : 0;
    noc_async_write(l1 + 256, TensorAccessor(a4, in_addr, 4).get_noc_addr(0), 4);
    noc_async_write_barrier();
    for (uint32_t i = 0; i < 32; ++i) {
        ix[i] = i < B ? p + i : p;
    }
    noc_async_write(l1 + 256, TensorAccessor(a5, ib_addr, 128).get_noc_addr(0), 128);
    noc_async_write_barrier();

    mask_row(TensorAccessor(a6, gm_addr, 4096), l1 + 512, SG, (int32_t)p, 0, W, B, NEG);
    mask_row(TensorAccessor(a7, sm_addr, 4096), l1 + 512, SW, (int32_t)p, 1, W, B, NEG);
    noc_async_write_barrier();
}
