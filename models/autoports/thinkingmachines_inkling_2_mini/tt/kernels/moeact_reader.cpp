// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Prefill MoE activation (tt/moeact.py), reader: row tiles [r0, r1) of the gate|up output [Rd, 2W] bf16; per row tile
// its routing-weight tile (column 0 of the [Rd, 1] weights, spread to every column here) and, per column tile c < Wt,
// the gate tile (r, c) and the up tile (r, Wt + c).
//
// runtime args: r0, r1      common args: gu, weights

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_g = get_compile_time_arg_val(0);
    constexpr uint32_t cb_u = get_compile_time_arg_val(1);
    constexpr uint32_t cb_w = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
    constexpr auto g_args = TensorAccessorArgs<4>();
    constexpr auto w_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    const uint32_t r0 = get_arg_val<uint32_t>(0), r1 = get_arg_val<uint32_t>(1);
    const auto ag = TensorAccessor(g_args, get_common_arg_val<uint32_t>(0), 2048);
    const auto aw = TensorAccessor(w_args, get_common_arg_val<uint32_t>(1), 2048);
    for (uint32_t r = r0; r < r1; ++r) {
        cb_reserve_back(cb_w, 1);
        const uint32_t wl = get_write_ptr(cb_w);
        noc_async_read(aw.get_noc_addr(r), wl, 2048);
        noc_async_read_barrier();
        // column 0 of each row (faces 0 / 2) -> all 32 columns (faces 0 / 1 and 2 / 3)
        volatile tt_l1_ptr uint16_t* t = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(wl);
        for (uint32_t h = 0; h < 2; ++h) {
            for (uint32_t row = 0; row < 16; ++row) {
                const uint32_t base = h * 512 + row * 16;  // (face 0 / 2: 256 values each, rows of 16)
                const uint16_t v = t[base];
                for (uint32_t col = 0; col < 16; ++col) {
                    t[base + col] = v;
                    t[base + 256 + col] = v;  // (face 1 / 3)
                }
            }
        }
        cb_push_back(cb_w, 1);
        for (uint32_t c = 0; c < Wt; ++c) {
            cb_reserve_back(cb_g, 1);
            cb_reserve_back(cb_u, 1);
            noc_async_read(ag.get_noc_addr(r * 2 * Wt + c), get_write_ptr(cb_g), 2048);
            noc_async_read(ag.get_noc_addr(r * 2 * Wt + Wt + c), get_write_ptr(cb_u), 2048);
            noc_async_read_barrier();
            cb_push_back(cb_g, 1);
            cb_push_back(cb_u, 1);
        }
    }
}
