// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Decode add + RMSNorm, writer: this core's sum-of-squares tile (faces 0 and 2, or face 0 only for <= 16 rows: column
// 0 holds the row sums) into slot `shard` of cb_slots on every core of the op, then one
// increment of each core's program semaphore; waits for NC increments (every core's tile), pushes the slots to the
// compute; then the normed tiles to pages shard * T + t of the interleaved output.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t shard = get_arg_val<uint32_t>(1);
    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr uint32_t NC = get_compile_time_arg_val(1);
    constexpr uint32_t SEM = get_compile_time_arg_val(2);
    constexpr uint32_t cb_p = get_compile_time_arg_val(3);
    constexpr uint32_t cb_slots = get_compile_time_arg_val(4);
    constexpr uint32_t cb_out = get_compile_time_arg_val(5);
    constexpr uint32_t FACES = get_compile_time_arg_val(6);  // 1: rows 0..15 only (face 0)
    constexpr uint32_t SEM_READY = get_compile_time_arg_val(7);
    constexpr auto out_args = TensorAccessorArgs<8>();
    constexpr uint32_t BF_TILE = 2048, F32_TILE = 4096, F32_FACE = 1024;
    const auto s_out = TensorAccessor(out_args, out_addr, BF_TILE);
    const uint32_t sem = get_semaphore(SEM);
    const uint32_t ready = get_semaphore(SEM_READY);
    // this core is in the program: tell every core (a core may start while others still run the previous program, whose
    // L1 the slot writes below would land in; semaphores are set up before any core of a program starts)
    for (uint32_t c = 0; c < NC; ++c) {
        const uint32_t x = get_arg_val<uint32_t>(2 + 2 * c), y = get_arg_val<uint32_t>(3 + 2 * c);
        noc_semaphore_inc(get_noc_addr(x, y, ready), 1);
    }

    DeviceZoneScopedN("wr");
    cb_reserve_back(cb_slots, NC);
    const uint32_t slots = get_write_ptr(cb_slots);  // the same address on every core of the program
    cb_wait_front(cb_p, 1);
    {
    DeviceZoneScopedN("xchg");
    volatile tt_l1_ptr uint32_t* ready_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready);
    noc_semaphore_wait(ready_ptr, NC);  // every core started: its slots are free
    noc_semaphore_set(ready_ptr, 0);
    const uint32_t p = get_read_ptr(cb_p);
    for (uint32_t c = 0; c < NC; ++c) {
        const uint32_t x = get_arg_val<uint32_t>(2 + 2 * c), y = get_arg_val<uint32_t>(3 + 2 * c);
        const uint32_t dst = slots + shard * F32_TILE;
        noc_async_write(p, get_noc_addr(x, y, dst), F32_FACE);
        if constexpr (FACES == 2) {
            noc_async_write(p + 2 * F32_FACE, get_noc_addr(x, y, dst + 2 * F32_FACE), F32_FACE);
        }
    }
    noc_async_write_barrier();
    for (uint32_t c = 0; c < NC; ++c) {
        const uint32_t x = get_arg_val<uint32_t>(2 + 2 * c), y = get_arg_val<uint32_t>(3 + 2 * c);
        noc_semaphore_inc(get_noc_addr(x, y, sem), 1);
    }
    cb_pop_front(cb_p, 1);
    volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem);
    noc_semaphore_wait(sem_ptr, NC);
    noc_semaphore_set(sem_ptr, 0);
    cb_push_back(cb_slots, NC);
    }

    cb_wait_front(cb_out, T);
    const uint32_t o = get_read_ptr(cb_out);
    for (uint32_t t = 0; t < T; ++t) {
        noc_async_write(o + t * BF_TILE, s_out.get_noc_addr(shard * T + t), BF_TILE);
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    cb_pop_front(cb_out, T);
}
