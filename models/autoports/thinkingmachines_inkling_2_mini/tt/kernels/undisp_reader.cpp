// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Merged MoE un-dispatch (tt/undisp.py), reader: per work item (token tile rt, column block cs), per sub-shard s and
// slot j, the 32 rows undispatch[t * K + j] of sub-shard s's down output (columns of block cs) as one row-major block.
// Tokens past T read as zeros.
//
// runtime args: item_start, item_count      common args: und, rows_0 .. rows_{S-1}

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t S = get_compile_time_arg_val(2);
    constexpr uint32_t Wt_c = get_compile_time_arg_val(3);
    constexpr uint32_t col_splits = get_compile_time_arg_val(4);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(5);  // Hc * 2
    constexpr uint32_t page_bytes = get_compile_time_arg_val(6);  // H * 2
    constexpr uint32_t cb_in = get_compile_time_arg_val(7);
    constexpr uint32_t cb_und = get_compile_time_arg_val(8);
    constexpr auto rows_args = TensorAccessorArgs<9>();
    constexpr auto und_args = TensorAccessorArgs<rows_args.next_compile_time_args_offset()>();

    const uint32_t item_start = get_arg_val<uint32_t>(0);
    const uint32_t item_count = get_arg_val<uint32_t>(1);
    const auto s_und = TensorAccessor(und_args, get_common_arg_val<uint32_t>(0), T * K * 4);
    const uint32_t und_l1 = get_write_ptr(cb_und);
    volatile tt_l1_ptr uint32_t* und = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(und_l1);

    for (uint32_t it = 0; it < item_count; ++it) {
        const uint32_t item = item_start + it;
        const uint32_t rt = item / col_splits;
        const uint32_t cs = item % col_splits;
        const uint32_t t0 = rt * 32;
        const uint32_t n_rows = (T - t0) < 32 ? (T - t0) : 32;
        noc_async_read(s_und.get_noc_addr(0) + t0 * K * 4, und_l1, n_rows * K * 4);
        noc_async_read_barrier();
        for (uint32_t s = 0; s < S; ++s) {
            const auto s_rows = TensorAccessor(rows_args, get_common_arg_val<uint32_t>(1 + s), page_bytes);
            for (uint32_t j = 0; j < K; ++j) {
                cb_reserve_back(cb_in, Wt_c);
                const uint32_t base = get_write_ptr(cb_in);
                for (uint32_t r = 0; r < n_rows; ++r) {
                    noc_async_read(s_rows.get_noc_addr(und[r * K + j]) + cs * row_bytes, base + r * row_bytes, row_bytes);
                }
                if (n_rows < 32) {
                    volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + n_rows * row_bytes);
                    for (uint32_t w = 0; w < (32 - n_rows) * row_bytes / 4; ++w) {
                        z[w] = 0;
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_in, Wt_c);
            }
        }
    }
}
