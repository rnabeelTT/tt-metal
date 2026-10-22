// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Down-projection activation gather + output scatter (BRISC, NOC1) of the fused MoE decode (moe_decode/experts.py).
//
// 1. in0 = the [k + 1, I] BF16 activation buffer (row s = w_s * silu(gate) * up of slot s, written by the gate|up op)
//    as (k + 1) * I / 32 1x32 tiles, slot-major: the K order of the stacked down weights in cb_w.
// 2. Writes each 1x32 output tile (output column n = bank + banks * (c0 + c)) into row 0 of tile n of the [32, H]
//    BF16 TILE output (16 values at face 0 row 0, 16 at face 1 row 0); rows 1..31 are never written (zero buffer).
//
// MOE_FUSED (the down cores of the one-program MoE, ExpertMoE): first the router top-k of logits [E / 2, E) for the
// reader (cb_part), then wait until every gate|up core has written its activations (program semaphore MOE_SEM counts
// MOE_NOTIFIERS increments) before reading them.
//
// runtime args: [act_addr, out_addr, bank, c0(, logits_addr)]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#ifdef MOE_FUSED
#include "router_topk.hpp"
#endif

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
    constexpr uint32_t cb_part = get_compile_time_arg_val(6);
    constexpr uint32_t cb_lg = get_compile_time_arg_val(7);
    constexpr uint32_t num_experts = get_compile_time_arg_val(8);
    constexpr auto act_args = TensorAccessorArgs<9>();
    constexpr auto out_args = TensorAccessorArgs<act_args.next_compile_time_args_offset()>();

    constexpr uint32_t kin = num_slots * act_row_bytes / 64;

    const auto s_act = TensorAccessor(act_args, act_addr, act_row_bytes);
    const auto s_out = TensorAccessor(out_args, out_addr, 2048);

#ifdef MOE_FUSED
    {
        constexpr auto lg_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
        constexpr uint32_t topk = num_slots - 1;
        const auto s_lg = TensorAccessor(lg_args, get_arg_val<uint32_t>(4), 2048);
        cb_reserve_back(cb_lg, 1);
        const uint32_t lg = get_write_ptr(cb_lg);
        for (uint32_t t = num_experts / 64; t < num_experts / 32; ++t) {
            noc_async_read(s_lg.get_noc_addr(t, 0), lg + t * 64, 32);
            noc_async_read(s_lg.get_noc_addr(t, 512), lg + t * 64 + 32, 32);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        cb_reserve_back(cb_part, 1);
        uint32_t* part = reinterpret_cast<uint32_t*>(get_write_ptr(cb_part));
        moe_topk::select_range<topk>(
            reinterpret_cast<const uint32_t*>(lg), num_experts / 4, num_experts / 2, part, part + topk);
        cb_push_back(cb_part, 1);
    }
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(MOE_SEM)), MOE_NOTIFIERS);
#endif
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
