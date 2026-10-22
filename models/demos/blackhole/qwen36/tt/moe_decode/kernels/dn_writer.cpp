// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Down-projection activation gather + output scatter (BRISC, NOC1) of the fused MoE decode (moe_decode/experts.py).
//
// 1. in0 = the [k + 1, I] BF16 activation buffer (row s = w_s * silu(gate) * up of slot s, written by the gate|up op)
//    as (k + 1) * I / 32 1x32 tiles, slot-major: the K order of the stacked down weights in cb_w.
// 2. Writes each 1x32 output tile (output column n = bank + banks * (c0 + c)) into row 0 of tile n of the [32, H]
//    BF16 TILE output (16 values at face 0 row 0, 16 at face 1 row 0); rows 1..31 are never written (zero buffer).
//
// runtime args: [act_addr, out_addr, bank, c0]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t act_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t bank = get_arg_val<uint32_t>(2);
    const uint32_t c0 = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t num_slots = get_compile_time_arg_val(2);
    constexpr uint32_t act_row_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t cols = get_compile_time_arg_val(4);
    constexpr uint32_t banks = get_compile_time_arg_val(5);
    constexpr auto act_args = TensorAccessorArgs<6>();
    constexpr auto out_args = TensorAccessorArgs<act_args.next_compile_time_args_offset()>();

    constexpr uint32_t kin = num_slots * act_row_bytes / 64;

    const auto s_act = TensorAccessor(act_args, act_addr, act_row_bytes);
    const auto s_out = TensorAccessor(out_args, out_addr, 2048);

    cb_reserve_back(cb_in0, kin);
    const uint32_t in0 = get_write_ptr(cb_in0);
    for (uint32_t s = 0; s < num_slots; ++s) {
        noc_async_read(s_act.get_noc_addr(s), in0 + s * act_row_bytes, act_row_bytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb_in0, kin);

    cb_wait_front(cb_out, cols);
    const uint32_t src = get_read_ptr(cb_out);
    for (uint32_t c = 0; c < cols; ++c) {
        const uint32_t n = bank + banks * (c0 + c);
        noc_async_write(src + c * 64, s_out.get_noc_addr(n, 0), 32);
        noc_async_write(src + c * 64 + 32, s_out.get_noc_addr(n, 512), 32);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, cols);
}
