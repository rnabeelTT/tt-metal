// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// DSpark drafter base logits (tt/drafter.py BaseHead; reader = qwen36 dl_reader, compute = head_compute without the
// softcap): rows 0 .. NR of the bf16 block hidden into cb_x, then rows 1 .. NR of every FP32 output column tile to the
// compact logits tensor [1, 8, NT, 32] FP32 ROW_MAJOR (page = (row, column tile), 128 B).
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
    constexpr uint32_t NR = get_compile_time_arg_val(3);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(4);
    constexpr auto x_args = TensorAccessorArgs<5>();
    constexpr auto o_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    static_assert(NR >= 1 && NR < 16, "rows 1..NR in faces 0 / 1");
    constexpr uint32_t row_bytes = (NR + 1) * 32;

    if (ncols == 0) {
        return;
    }
    cb_reserve_back(cb_x, kt);
    const uint32_t xl1 = get_write_ptr(cb_x);
    const auto s_x = TensorAccessor(x_args, x_addr, 2048);
    for (uint32_t k = 0; k < kt; ++k) {
        const uint64_t src = s_x.get_noc_addr(k);
        noc_async_read(src, xl1 + k * 2048, row_bytes);
        noc_async_read(src + 512, xl1 + k * 2048 + 512, row_bytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, kt);

    const auto s_o = TensorAccessor(o_args, out_addr, 128);
    for (uint32_t i = 0; i < ncols; ++i) {
        cb_wait_front(cb_out, 1);
        const uint32_t col = out_col0 + i;
        if (col < n_tiles) {
            const uint32_t t = get_read_ptr(cb_out);
            for (uint32_t r = 1; r <= NR; ++r) {
                const uint64_t dst = s_o.get_noc_addr(r * n_tiles + col);
                noc_async_write(t + r * 64, dst, 64);
                noc_async_write(t + 1024 + r * 64, dst + 64, 64);
            }
            noc_async_writes_flushed();
        }
        cb_pop_front(cb_out, 1);
    }
    noc_async_write_barrier();
}
