// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode layer boundary, receiver (NCRISC) (fabric_boundary/ops.py: FabricBoundary). From the gpt-oss fused decode
// receiver (models/demos/gpt_oss/tt/fused_decode/kernels/boundary_reader.cpp), plus the entry gather and the notify.
//
// Feeds the boundary compute: the RMSNorm weight (`tiles` BF16 pages, one DRAM page) and the reduce scaler, the
// residual (cb_res) and, once every device's partial sum has arrived (the receive semaphore reaches `expected`: the
// local copy + one packet per remote device), the `slots` x `tiles` receive buffer (cb_recv). The semaphore is reset
// for the next use of this site before the slots are published. Then it waits for the normed output (cb_x) and
// increments program semaphore 0 on every to-rows core (fb_rows.cpp).
//
// residual source (res_mode):
//   0: cb_res is backed by the flat residual tensor (published as is);
//   2: entry: row 0 of each 32x32 BF16 tile of a [1, 1, 32, H] tensor (the gathered embedding rows), packed flat.
//
// runtime args: [gamma_addr, sem_addr, res_src_addr, n_rows_cores, (noc x, noc y) per to-rows core]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t gamma_addr = get_arg_val<uint32_t>(0);
    const uint32_t sem_addr = get_arg_val<uint32_t>(1);
    const uint32_t res_src_addr = get_arg_val<uint32_t>(2);
    const uint32_t n_rows_cores = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_res = get_compile_time_arg_val(0);
    constexpr uint32_t cb_recv = get_compile_time_arg_val(1);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(3);
    constexpr uint32_t tiles = get_compile_time_arg_val(4);
    constexpr uint32_t slots = get_compile_time_arg_val(5);
    constexpr uint32_t expected = get_compile_time_arg_val(6);
    constexpr uint32_t res_mode = get_compile_time_arg_val(7);
    constexpr uint32_t cb_x = get_compile_time_arg_val(8);
    constexpr uint32_t ntiles = get_compile_time_arg_val(9);  // 32x32 tiles of the entry source
    constexpr auto gamma_args = TensorAccessorArgs<10>();
    constexpr auto res_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    constexpr uint32_t page = 2048;

    const auto s_gamma = TensorAccessor(gamma_args, gamma_addr, tiles * page);
    cb_reserve_back(cb_gamma, tiles);
    noc_async_read(s_gamma.get_noc_addr(0), get_write_ptr(cb_gamma), tiles * page);

    cb_reserve_back(cb_res, tiles);
    if constexpr (res_mode == 2) {
        const auto s_res = TensorAccessor(res_args, res_src_addr, page);
        const uint32_t dst = get_write_ptr(cb_res);
        for (uint32_t t = 0; t < ntiles; ++t) {
            const uint64_t src = s_res.get_noc_addr(t);
            noc_async_read(src, dst + 64 * t, 32);
            noc_async_read(src + 512, dst + 64 * t + 32, 32);
        }
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t z = 64 * ntiles; z < tiles * page; z += MEM_ZEROS_SIZE) {
            const uint32_t n = tiles * page - z < MEM_ZEROS_SIZE ? tiles * page - z : MEM_ZEROS_SIZE;
            noc_async_read(zeros, dst + z, n);
        }
    }

    // Reduce scaler: 1.0 in the first row of each face of a BF16 tile (scalar sum of all 1024 values).
    cb_reserve_back(cb_scaler, 1);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_scaler));
    for (uint32_t i = 0; i < page / 4; ++i) {
        s[i] = 0;
    }
    for (uint32_t f = 0; f < 4; ++f) {
        for (uint32_t i = 0; i < 8; ++i) {
            s[f * 128 + i] = 0x3F803F80;
        }
    }
    cb_push_back(cb_scaler, 1);

    noc_async_read_barrier();
    cb_push_back(cb_res, tiles);
    cb_push_back(cb_gamma, tiles);

    if constexpr (slots > 0) {
        {
            DeviceZoneScopedN("FB_WAIT");
            volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
            noc_semaphore_wait_min(sem, expected);
            noc_semaphore_set(sem, 0);
        }
        cb_reserve_back(cb_recv, slots * tiles);
        cb_push_back(cb_recv, slots * tiles);
    }

    cb_wait_front(cb_x, tiles);
    const uint32_t sem0 = get_semaphore(0);
    for (uint32_t i = 0; i < n_rows_cores; ++i) {
        noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(4 + 2 * i), get_arg_val<uint32_t>(5 + 2 * i), sem0), 1);
    }
    noc_async_atomic_barrier();
    cb_pop_front(cb_x, tiles);
}
