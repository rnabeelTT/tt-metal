// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// GDN decode head step, writer: the unit's new state tiles back in place ([Bmax, Nv, Dk, Dv] bf16 TILE, DRAM) and its
// gated output row into row b of the out-proj input ([1, B, Nv * Dv] bf16 TILE, L1): two 32-byte face-row writes per
// tile, disjoint from every other unit's (no read-modify-write).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t state_addr = get_arg_val<uint32_t>(0);
    const uint32_t y_addr = get_arg_val<uint32_t>(1);
    const uint32_t start = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);

    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t Nv = get_compile_time_arg_val(2);
    constexpr uint32_t cb_state = get_compile_time_arg_val(3);
    constexpr uint32_t cb_y = get_compile_time_arg_val(4);
    constexpr auto state_args = TensorAccessorArgs<5>();
    constexpr auto y_args = TensorAccessorArgs<state_args.next_compile_time_args_offset()>();
    constexpr uint32_t kv = Kt * Vt;
    constexpr uint32_t BF_TILE = 2048, BF_FACE = 512, BF_ROW = 32;

    const auto s_state = TensorAccessor(state_args, state_addr, BF_TILE);
    const auto s_y = TensorAccessor(y_args, y_addr, BF_TILE);

    for (uint32_t w = 0; w < count; ++w) {
        const uint32_t n = start + w;
        const uint32_t b = n / Nv;
        const uint32_t h = n - b * Nv;
        cb_wait_front(cb_state, kv);
        const uint32_t s = get_read_ptr(cb_state);
        for (uint32_t i = 0; i < kv; ++i) {
            noc_async_write(s + i * BF_TILE, s_state.get_noc_addr(n * kv + i), BF_TILE);
        }
        cb_wait_front(cb_y, Vt);
        const uint32_t y = get_read_ptr(cb_y);
        const uint32_t r = b & 31;
        for (uint32_t j = 0; j < Vt; ++j) {
            const uint32_t page = (b >> 5) * (Nv * Vt) + h * Vt + j;
            for (uint32_t c = 0; c < 2; ++c) {
                const uint32_t off = ((r >> 4) * 2 + c) * BF_FACE + (r & 15) * BF_ROW;
                noc_async_write(y + j * BF_TILE + c * BF_FACE, s_y.get_noc_addr(page, off), BF_ROW);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_state, kv);
        cb_pop_front(cb_y, Vt);
    }
}
