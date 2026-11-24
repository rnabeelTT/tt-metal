// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Inkling short conv, commit: state rows := seq[n .. n + 2] (the 3 inputs ending at accepted row n - 1) per
// 32-channel column, from the seq tile the rows pass wrote (bf16 TILE) into the fp32 row-major state [3, C].

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "tile_rows.hpp"

void kernel_main() {
    const uint32_t seq_addr = get_arg_val<uint32_t>(0);
    const uint32_t st_addr = get_arg_val<uint32_t>(1);
    const uint32_t c0 = get_arg_val<uint32_t>(2);
    const uint32_t n = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(0);
    constexpr uint32_t cols = get_compile_time_arg_val(1);
    constexpr uint32_t shift = get_compile_time_arg_val(2);
    constexpr uint32_t TB = 2048;
    constexpr uint32_t ROWB = cols * 128;
    constexpr auto seq_args = TensorAccessorArgs<3>();
    constexpr auto st_args = TensorAccessorArgs<seq_args.next_compile_time_args_offset()>();
    const auto s_seq = TensorAccessor(seq_args, seq_addr, TB);
    const auto s_st = TensorAccessor(st_args, st_addr, ROWB);
    const uint32_t tile = get_write_ptr(cb_scr);
    const uint32_t rows = tile + TB;
    volatile uint32_t* tw = reinterpret_cast<volatile uint32_t*>(tile);
    volatile uint32_t* rw = reinterpret_cast<volatile uint32_t*>(rows);
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t c = c0 + i;
        noc_async_read(s_seq.get_noc_addr(c), tile, TB);
        noc_async_read_barrier();
        for (uint32_t r = 0; r < 3; ++r) {
            sconv::row_to_f32(rw + r * 32, tw, shift + r);
        }
        for (uint32_t r = 0; r < 3; ++r) {
            noc_async_write(rows + r * 128, s_st.get_noc_addr(r, c * 128), 128);
        }
        noc_async_write_barrier();
    }
}
