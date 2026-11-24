// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Message gate in one program (tt/gate.py), writer: the two cross-core rounds and the output.
//   start: every core increments the root's `ready`; the root waits for all C, then increments every core's `go`
//          (no core writes into another core's L1 before that core's program runs);
//   round 1: each core writes its 2 partial row-sum tiles (sum h^2, sum m^2) into slot c of the root's cb_gath and
//          increments the root's s1; the root's compute reduces them to rsqrt factors (cb_res1), which the root writes
//          into every core's cb_r (then increments their sr);
//   round 2: the same with the R partial down-projection tiles (cb_part2 -> cb_gath; cb_res2 -> every core's cb_g);
//   then this core's PER output tiles.
// The CBs sit at the same L1 address on every core of the program; a fresh (or fully popped) CB starts at its base.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t t0 = get_arg_val<uint32_t>(1);
    const uint32_t c = get_arg_val<uint32_t>(2);  // this core's slot (0 = root)
    const uint32_t root_x = get_arg_val<uint32_t>(3);
    const uint32_t root_y = get_arg_val<uint32_t>(4);
    // args 5.. : the NOC x, y of every core (root only)

    constexpr uint32_t cb_part1 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gath = get_compile_time_arg_val(1);
    constexpr uint32_t cb_res1 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_r = get_compile_time_arg_val(3);
    constexpr uint32_t cb_part2 = get_compile_time_arg_val(4);
    constexpr uint32_t cb_res2 = get_compile_time_arg_val(5);
    constexpr uint32_t cb_g = get_compile_time_arg_val(6);
    constexpr uint32_t cb_out = get_compile_time_arg_val(7);
    constexpr uint32_t C = get_compile_time_arg_val(8);
    constexpr uint32_t PER = get_compile_time_arg_val(9);
    constexpr uint32_t R = get_compile_time_arg_val(10);
    constexpr uint32_t F32 = 4096, BF = 2048;
    constexpr auto o_args = TensorAccessorArgs<11>();
    const auto ao = TensorAccessor(o_args, out_addr, F32);

    const uint32_t s_ready = get_semaphore(0), s_go = get_semaphore(1), s1 = get_semaphore(2);
    const uint32_t sr = get_semaphore(3), s2 = get_semaphore(4), sg = get_semaphore(5);
    auto sem = [](uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a); };
    const bool root = c == 0;
    auto px = [](uint32_t j) { return get_arg_val<uint32_t>(5 + 2 * j); };
    auto py = [](uint32_t j) { return get_arg_val<uint32_t>(6 + 2 * j); };

    // receive buffers: reserved now (their addresses are where the root writes)
    cb_reserve_back(cb_r, 2);
    cb_reserve_back(cb_g, R);
    if (root) {
        cb_reserve_back(cb_gath, 2 * C);
    }
    const uint32_t gath = get_write_ptr(cb_gath);  // (the same address on every core: not yet used)
    const uint32_t r_addr = get_write_ptr(cb_r);
    const uint32_t g_addr = get_write_ptr(cb_g);

    // start handshake
    {
    DeviceZoneScopedN("G_W_START");
    noc_semaphore_inc(get_noc_addr(root_x, root_y, s_ready), 1);
    if (root) {
        noc_semaphore_wait(sem(s_ready), C);
        for (uint32_t j = 0; j < C; ++j) {
            noc_semaphore_inc(get_noc_addr(px(j), py(j), s_go), 1);
        }
    }
    noc_semaphore_wait(sem(s_go), 1);
    }

    {
    DeviceZoneScopedN("G_W_R1");
    cb_wait_front(cb_part1, 2);
    noc_async_write(get_read_ptr(cb_part1), get_noc_addr(root_x, root_y, gath + c * 2 * F32), 2 * F32);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(root_x, root_y, s1), 1);
    cb_pop_front(cb_part1, 2);
    if (root) {
        noc_semaphore_wait(sem(s1), C);
        cb_push_back(cb_gath, 2 * C);
        cb_wait_front(cb_res1, 2);
        for (uint32_t j = 0; j < C; ++j) {
            noc_async_write(get_read_ptr(cb_res1), get_noc_addr(px(j), py(j), r_addr), 2 * F32);
        }
        noc_async_write_barrier();
        for (uint32_t j = 0; j < C; ++j) {
            noc_semaphore_inc(get_noc_addr(px(j), py(j), sr), 1);
        }
        cb_pop_front(cb_res1, 2);
        cb_reserve_back(cb_gath, R * C);  // the compute popped round 1 before producing cb_res1: back at the base
    }
    noc_semaphore_wait(sem(sr), 1);
    cb_push_back(cb_r, 2);
    }

    {
    DeviceZoneScopedN("G_W_R2");
    cb_wait_front(cb_part2, R);
    noc_async_write(get_read_ptr(cb_part2), get_noc_addr(root_x, root_y, gath + c * R * F32), R * F32);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(root_x, root_y, s2), 1);
    cb_pop_front(cb_part2, R);
    if (root) {
        noc_semaphore_wait(sem(s2), C);
        cb_push_back(cb_gath, R * C);
        cb_wait_front(cb_res2, R);
        for (uint32_t j = 0; j < C; ++j) {
            noc_async_write(get_read_ptr(cb_res2), get_noc_addr(px(j), py(j), g_addr), R * BF);
        }
        noc_async_write_barrier();
        for (uint32_t j = 0; j < C; ++j) {
            noc_semaphore_inc(get_noc_addr(px(j), py(j), sg), 1);
        }
        cb_pop_front(cb_res2, R);
    }
    noc_semaphore_wait(sem(sg), 1);
    cb_push_back(cb_g, R);
    }

    // output
    for (uint32_t t = 0; t < PER; ++t) {
        cb_wait_front(cb_out, 1);
        noc_async_write(get_read_ptr(cb_out), ao.get_noc_addr(t0 + t), F32);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
    // semaphores back to 0 for the next launch (every increment aimed at this core has landed by now)
    *sem(s_go) = 0;
    *sem(sr) = 0;
    *sem(sg) = 0;
    if (root) {
        *sem(s_ready) = 0;
        *sem(s1) = 0;
        *sem(s2) = 0;
    }
}
