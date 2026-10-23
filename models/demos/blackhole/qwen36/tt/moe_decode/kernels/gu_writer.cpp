// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Activation gather + SwiGLU output scatter (BRISC, NOC1) of the fused MoE decode gate|up (moe_decode/experts.py).
//
// 1. Reads row 0 of the normed hidden x [32, H] (BF16 TILE, interleaved or sharded): per tile, face 0 row 0 and face 1
//    row 0 = one 1x32 activation tile (64 bytes) of cb_x.
// 2. Router tail, upper half: reads the logits row and hands the reader the top-k of logits [E / 2, E) (cb_part:
//    k order keys, k ids).
// 3. Waits for the reader's route record (cb_route: k ids, then the k + 1 slot weights) and hands this core's slot
//    weights to the compute MATH thread through the TRISC mailbox (FP32 bits). The route-owner core also writes the
//    record to the route buffer the down op reads ([ids | weights] UINT16 row-major, 64 bytes).
// 4. Writes each 1x32 output tile (w_s * silu(gate) * up of slot s and column pair p, columns 32 p .. 32 p + 31; the
//    group's pairs as gu_layout.hpp) to row s of the [k + 1, I] BF16 row-major activation buffer.
//
// runtime args: [x_addr, act_addr, route_addr, logits_addr, group q, route_owner, num_slots, slot...,
//                (ROUTER_SIGB) bias_addr, (MOE_FUSED) the down cores]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "gu_layout.hpp"
#include "router_topk.hpp"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t act_addr = get_arg_val<uint32_t>(1);
    const uint32_t route_addr = get_arg_val<uint32_t>(2);
    const uint32_t logits_addr = get_arg_val<uint32_t>(3);
    const uint32_t pair = get_arg_val<uint32_t>(4);
    const uint32_t route_owner = get_arg_val<uint32_t>(5);
    const uint32_t num_slots = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_act = get_compile_time_arg_val(1);
    constexpr uint32_t cb_route = get_compile_time_arg_val(2);
    constexpr uint32_t kx = get_compile_time_arg_val(3);  // H / 32
    constexpr uint32_t topk = get_compile_time_arg_val(4);
    constexpr uint32_t num_experts = get_compile_time_arg_val(5);
    constexpr uint32_t act_row_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t cb_part = get_compile_time_arg_val(7);
    constexpr uint32_t cb_lg = get_compile_time_arg_val(8);
    constexpr uint32_t m = get_compile_time_arg_val(9);  // up|gate columns per bank and expert
    constexpr uint32_t banks = get_compile_time_arg_val(10);
    constexpr uint32_t np = gu_layout::pairs<m>();
    constexpr auto x_args = TensorAccessorArgs<11>();
    constexpr auto act_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto route_args = TensorAccessorArgs<act_args.next_compile_time_args_offset()>();
    constexpr auto lg_args = TensorAccessorArgs<route_args.next_compile_time_args_offset()>();

    constexpr uint32_t tiny_bytes = 64;
    constexpr uint32_t lg_tiles = num_experts / 32;  // the experts' logits (not the shared-gate column)

    const auto s_x = TensorAccessor(x_args, x_addr, 2048);
    const auto s_act = TensorAccessor(act_args, act_addr, act_row_bytes);
    const auto s_route = TensorAccessor(route_args, route_addr, 64);
    const auto s_lg = TensorAccessor(lg_args, logits_addr, moe_topk::LG_PAGE);

    // logits [E / 2, E) (they gate the expert stream)
    cb_reserve_back(cb_lg, 1);
    const uint32_t lg = get_write_ptr(cb_lg);
    for (uint32_t t = lg_tiles / 2; t < lg_tiles; ++t) {
        noc_async_read(s_lg.get_noc_addr(t, 0), lg + t * moe_topk::LG_TILE_L1, moe_topk::LG_HALF);
        noc_async_read(
            s_lg.get_noc_addr(t, moe_topk::LG_FACE1), lg + t * moe_topk::LG_TILE_L1 + moe_topk::LG_HALF, moe_topk::LG_HALF);
    }
    const int32_t* bias = nullptr;
#ifdef ROUTER_SIGB
    constexpr uint32_t extra = 1;  // the bias address after the slot list
    const uint32_t bias_l1 = lg + lg_tiles * moe_topk::LG_TILE_L1;
    noc_async_read(get_noc_addr_from_bank_id<true>(0, get_arg_val<uint32_t>(7 + num_slots)), bias_l1, 4 * num_experts);
    bias = reinterpret_cast<const int32_t*>(bias_l1);
#else
    constexpr uint32_t extra = 0;
#endif
    noc_async_read_barrier();
    invalidate_l1_cache();

    {
        DeviceZoneScopedN("tail");
        cb_reserve_back(cb_part, 1);
        uint32_t* part = reinterpret_cast<uint32_t*>(get_write_ptr(cb_part));
        moe_topk::route_select<topk>(lg, bias, num_experts / 2, num_experts, part, part + topk);
        cb_push_back(cb_part, 1);
    }

    // x after the router tail: the compute needs it only with the first weight column
    cb_reserve_back(cb_x, kx);
    const uint32_t x_l1 = get_write_ptr(cb_x);
    for (uint32_t t = 0; t < kx; ++t) {
        noc_async_read(s_x.get_noc_addr(t, 0), x_l1 + t * tiny_bytes, 32);
        noc_async_read(s_x.get_noc_addr(t, 512), x_l1 + t * tiny_bytes + 32, 32);
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, kx);

    cb_wait_front(cb_route, 1);
    const uint32_t route_l1 = get_read_ptr(cb_route);
    volatile tt_l1_ptr uint16_t* route = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(route_l1);
    for (uint32_t i = 0; i < num_slots; ++i) {
        const uint32_t s = get_arg_val<uint32_t>(7 + i);
        ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, static_cast<uint32_t>(route[16 + s]) << 16);
    }
    if (route_owner) {
        noc_async_write(route_l1, s_route.get_noc_addr(0), 64);
    }

    for (uint32_t i = 0; i < num_slots; ++i) {
        const uint32_t s = get_arg_val<uint32_t>(7 + i);
        cb_wait_front(cb_act, np);
        for (uint32_t p = 0; p < np; ++p) {
            noc_async_write(
                get_read_ptr(cb_act) + p * tiny_bytes,
                s_act.get_noc_addr(s, gu_layout::pair_index<m, banks>(pair, p) * tiny_bytes),
                tiny_bytes);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_act, np);
    }
    noc_async_write_barrier();
#ifdef MOE_FUSED
    // the one-program MoE: tell every down core this core's activations are written
    // (runtime args after the slot list: num_down, then (x, y) NOC coordinates per down core)
    const uint32_t a0 = 7 + num_slots + extra;
    const uint32_t num_down = get_arg_val<uint32_t>(a0);
    const uint32_t sem = get_semaphore(MOE_SEM);
    for (uint32_t d = 0; d < num_down; ++d) {
        noc_semaphore_inc(
            get_noc_addr(get_arg_val<uint32_t>(a0 + 1 + 2 * d), get_arg_val<uint32_t>(a0 + 2 + 2 * d), sem), 1);
    }
    noc_async_atomic_barrier();
#endif
    cb_pop_front(cb_route, 1);
}
