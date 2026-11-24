// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Batched short conv step (tt/bconv.py), writer: y (bf16) and the rotated fp32 states, in place (each channel tile is
// this core's alone: the reader read the old states before the compute released these tiles).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// BCONV_LIVE: rows 0 .. LIVE-1 only (faces 0 / 1)
#ifdef BCONV_LIVE
#define WR(src, dst, bytes) wr_live<bytes>(src, dst)
template <uint32_t BYTES>
inline void wr_live(uint32_t src, uint64_t dst) {
    constexpr uint32_t face = BYTES / 4, n = BCONV_LIVE * (BYTES / 64);
    noc_async_write(src, dst, n);
    noc_async_write(src + face, dst + face, n);
}
#else
#define WR(src, dst, bytes) noc_async_write(src, dst, bytes)
#endif


void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t s0_addr = get_arg_val<uint32_t>(1);
    const uint32_t s1_addr = get_arg_val<uint32_t>(2);
    const uint32_t s2_addr = get_arg_val<uint32_t>(3);
    const uint32_t t0 = get_arg_val<uint32_t>(4);
    const uint32_t nt = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_y = get_compile_time_arg_val(0);
    constexpr uint32_t cb_o0 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_o1 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_o2 = get_compile_time_arg_val(3);
    constexpr uint32_t BF16 = 2048, F32 = 4096;
    constexpr auto y_args = TensorAccessorArgs<4>();
    constexpr auto st_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    const auto ay = TensorAccessor(y_args, y_addr, BF16);
    const auto as0 = TensorAccessor(st_args, s0_addr, F32);
    const auto as1 = TensorAccessor(st_args, s1_addr, F32);
    const auto as2 = TensorAccessor(st_args, s2_addr, F32);

    for (uint32_t t = t0; t < t0 + nt; ++t) {
        cb_wait_front(cb_y, 1);
        cb_wait_front(cb_o0, 1);
        cb_wait_front(cb_o1, 1);
        cb_wait_front(cb_o2, 1);
        WR(get_read_ptr(cb_y), ay.get_noc_addr(t), BF16);
        WR(get_read_ptr(cb_o0), as0.get_noc_addr(t), F32);
        WR(get_read_ptr(cb_o1), as1.get_noc_addr(t), F32);
        WR(get_read_ptr(cb_o2), as2.get_noc_addr(t), F32);
        noc_async_write_barrier();
        cb_pop_front(cb_y, 1);
        cb_pop_front(cb_o0, 1);
        cb_pop_front(cb_o1, 1);
        cb_pop_front(cb_o2, 1);
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
