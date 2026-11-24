// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fused packed decode all-reduce (tt/ccl.py _Packed.fused), shard core: waits for this die's slot (its packers'
// increments of program semaphore 1) and the other dies' slots (one fused fabric write + increment of the site's
// global semaphore per die), then hands the N slots to the add (arfuse_add.cpp).
//
// common args: global semaphore address; runtime args: this core's packer count

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t KJ = get_compile_time_arg_val(1);
    constexpr uint32_t cb_recv = get_compile_time_arg_val(2);
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_common_arg_val<uint32_t>(0));
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1)), get_arg_val<uint32_t>(0));
    noc_semaphore_wait(sem, N - 1);
    noc_semaphore_set(sem, 0);
    cb_reserve_back(cb_recv, N * KJ);
    cb_push_back(cb_recv, N * KJ);
}
