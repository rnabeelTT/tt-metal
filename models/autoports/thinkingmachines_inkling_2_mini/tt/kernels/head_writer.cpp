// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Inkling LM head, activation in + logits out (BRISC; tt/head.py StreamHead): rows 0 .. rows-1 of the kt BF16
// activation tiles into cb_x (rows <= 16: faces 0 / 1, the other rows are don't-care; else whole tiles), then each FP32 output column tile the
// compute packs to logits page out_col0 + i (columns past n_tiles: bank padding, not written).
//
// runtime args: [x_addr, out_addr, out_col0, ncols]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_col0 = get_arg_val<uint32_t>(2);
    const uint32_t ncols = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t kt = get_compile_time_arg_val(2);
    constexpr uint32_t rows = get_compile_time_arg_val(3);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(4);
    constexpr auto x_args = TensorAccessorArgs<5>();
    constexpr auto o_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    static_assert(rows <= 32, "one tile row");
    constexpr uint32_t row_bytes = rows * 32;
    constexpr uint32_t OB = 4096;  // an FP32 tile

    if (ncols == 0) {
        return;
    }
    cb_reserve_back(cb_x, kt);
    const uint32_t xl1 = get_write_ptr(cb_x);
    const auto s_x = TensorAccessor(x_args, x_addr, 2048);
    for (uint32_t k = 0; k < kt; ++k) {
        const uint64_t src = s_x.get_noc_addr(k);
        if constexpr (rows > 16) {  // (all four faces)
            noc_async_read(src, xl1 + k * 2048, 2048);
        } else {
            noc_async_read(src, xl1 + k * 2048, row_bytes);
            noc_async_read(src + 512, xl1 + k * 2048 + 512, row_bytes);
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, kt);

    const auto s_o = TensorAccessor(o_args, out_addr, OB);
    for (uint32_t i = 0; i < ncols; ++i) {
        cb_wait_front(cb_out, 1);
        if (out_col0 + i < n_tiles) {
            noc_async_write(get_read_ptr(cb_out), s_o.get_noc_addr(out_col0 + i), OB);
            noc_async_writes_flushed();
        }
        cb_pop_front(cb_out, 1);
    }
    noc_async_write_barrier();
}
