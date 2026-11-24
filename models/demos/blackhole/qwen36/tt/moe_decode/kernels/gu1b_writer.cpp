// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// One-row activation gather + gate exchange + SwiGLU output scatter (BRISC, NOC1) of the one-bank gate|up op
// (moe_decode/experts.py ExpertGateUp1B, gu1b_compute.cpp).
//
// 1. Reads row 0 of the normed hidden x [32, H] (BF16 TILE): per tile, face 0 row 0 and face 1 row 0 = one 1x32
//    activation tile (64 bytes) of cb_x (as gu_writer.cpp).
// 2. Takes the route record from the reader (cb_route). Per slot this core owns, in order: hands the slot weight to the
//    compute MATH thread (TRISC mailbox, FP32 bits), then exchanges gate columns with the partner core (bank b +- banks
//    / 2, same rank, same slots: gur_writer.cpp GU_ONE_BANK): this core's gate-column FP32 tiles (cb_xo) go into the
//    partner's cb_xi ring (`xring` slots of nmax tiles at the same L1 address on both cores; semaphore 0: credits from
//    the partner, 1: slots that arrived from it), then the previous slot's output tiles to the activation buffer: one
//    1x32 tile per local up column c (pair b + banks c) to row s of the [k + 1, I] BF16 row-major buffer.
//
// runtime args: [x_addr, act_addr, bank, partner noc x, partner noc y, num_slots, slot...]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include "ckernel_defs.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t act_addr = get_arg_val<uint32_t>(1);
    const uint32_t bank = get_arg_val<uint32_t>(2);
    const uint32_t ppx = get_arg_val<uint32_t>(3);
    const uint32_t ppy = get_arg_val<uint32_t>(4);
    const uint32_t num_slots = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_act = get_compile_time_arg_val(1);
    constexpr uint32_t cb_route = get_compile_time_arg_val(2);
    constexpr uint32_t kx = get_compile_time_arg_val(3);  // H / 32
    constexpr uint32_t act_row_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t m = get_compile_time_arg_val(5);
    constexpr uint32_t banks = get_compile_time_arg_val(6);
    constexpr uint32_t cb_xo = get_compile_time_arg_val(7);
    constexpr uint32_t cb_xi = get_compile_time_arg_val(8);
    constexpr uint32_t xt = get_compile_time_arg_val(9);  // one FP32 tile
    constexpr uint32_t xring = get_compile_time_arg_val(10);
    constexpr auto x_args = TensorAccessorArgs<11>();
    constexpr auto act_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
#ifdef ROLE_HI
    constexpr uint32_t nup = (m - 1) / 2;
#else
    constexpr uint32_t nup = (m + 1) / 2;
#endif
    constexpr uint32_t nsend = m - nup;
    constexpr uint32_t nmax = (m + 1) / 2;
    static_assert(m % 2 == 1, "one-bank layout: odd m");
    constexpr uint32_t tiny_bytes = 64;

    const auto s_x = TensorAccessor(x_args, x_addr, 2048);
    const auto s_act = TensorAccessor(act_args, act_addr, act_row_bytes);

    // the partner's ring: credits for all of it (its writer sends there once it holds a credit)
    const uint32_t x_ring_base = get_write_ptr(cb_xi);
    volatile tt_l1_ptr uint32_t* x_credit = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    volatile tt_l1_ptr uint32_t* x_got = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    cb_reserve_back(cb_xi, xring * nmax);
    noc_semaphore_inc(get_noc_addr(ppx, ppy, get_semaphore(0)), xring);

    cb_reserve_back(cb_x, kx);
    const uint32_t x_l1 = get_write_ptr(cb_x);
    for (uint32_t t = 0; t < kx; ++t) {
        noc_async_read(s_x.get_noc_addr(t, 0), x_l1 + t * tiny_bytes, 32);
        noc_async_read(s_x.get_noc_addr(t, 512), x_l1 + t * tiny_bytes + 32, 32);
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, kx);

    cb_wait_front(cb_route, 1);
    volatile tt_l1_ptr uint16_t* route = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(cb_route));

    auto exchange = [&](uint32_t i) {
        if (i >= xring) {
            cb_reserve_back(cb_xi, nmax);  // slot i - xring consumed
            noc_semaphore_inc(get_noc_addr(ppx, ppy, get_semaphore(0)), 1);
        }
        cb_wait_front(cb_xo, nsend);
        noc_semaphore_wait_min(x_credit, i + 1);
        noc_async_write(get_read_ptr(cb_xo), get_noc_addr(ppx, ppy, x_ring_base + (i % xring) * nmax * xt), nsend * xt);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(ppx, ppy, get_semaphore(1)), 1);
        cb_pop_front(cb_xo, nsend);
        noc_semaphore_wait_min(x_got, i + 1);
        cb_push_back(cb_xi, nmax);
    };
    auto drain = [&](uint32_t s) {
        cb_wait_front(cb_act, nup);
        for (uint32_t c = 0; c < nup; ++c) {
            noc_async_write(get_read_ptr(cb_act) + c * tiny_bytes, s_act.get_noc_addr(s, (bank + banks * c) * tiny_bytes),
                            tiny_bytes);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_act, nup);
    };
    for (uint32_t i = 0; i < num_slots; ++i) {
        const uint32_t s = get_arg_val<uint32_t>(6 + i);
        ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, static_cast<uint32_t>(route[16 + s]) << 16);
        exchange(i);
        if (i > 0) {
            drain(get_arg_val<uint32_t>(6 + i - 1));
        }
    }
    if (num_slots > 0) {
        drain(get_arg_val<uint32_t>(6 + num_slots - 1));
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    cb_pop_front(cb_route, 1);
}
