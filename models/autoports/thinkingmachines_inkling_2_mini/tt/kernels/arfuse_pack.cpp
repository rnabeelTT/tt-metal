// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fused packed decode all-reduce (tt/ccl.py _Packed.fused), packer p: tiles [p TP, (p + 1) TP) of the T live rows of
// a [1, 1, 32, H] bf16 TILE tensor, packed (arpack.cpp's layout: chunk (2 t + f) T + r) into the staging CB, then
// written to this die's slot on the owning shard core(s) and the same place in its link's sender payload; then one
// increment per touched shard core (program semaphore 1) and per touched link sender (program semaphore 0).
//
// common args: tensor address, slot address, K, links, K shard (noc x, noc y), links sender (noc x, noc y)

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr uint32_t TP = get_compile_time_arg_val(1);
    constexpr uint32_t HT = get_compile_time_arg_val(2);
    constexpr uint32_t CX = get_compile_time_arg_val(3);
    constexpr uint32_t KPL = get_compile_time_arg_val(4);  // shard tiles per link
    constexpr uint32_t cb_stage = get_compile_time_arg_val(5);
    constexpr uint32_t cb_tx = get_compile_time_arg_val(6);
    constexpr auto t_args = TensorAccessorArgs<7>();
    const uint32_t p = (uint32_t)get_absolute_logical_y() * CX + (uint32_t)get_absolute_logical_x();
    const uint32_t t0 = p * TP;
    if (t0 >= HT) {
        return;
    }
    const uint32_t t1 = t0 + TP < HT ? t0 + TP : HT;
    const auto at = TensorAccessor(t_args, get_common_arg_val<uint32_t>(0), 2048);
    const uint32_t slot = get_common_arg_val<uint32_t>(1);
    const uint32_t K = get_common_arg_val<uint32_t>(2);
    auto shard_noc = [&](uint32_t k, uint32_t addr) {
        return get_noc_addr(get_common_arg_val<uint32_t>(4 + 2 * k), get_common_arg_val<uint32_t>(5 + 2 * k), addr);
    };
    auto send_noc = [&](uint32_t l, uint32_t addr) {
        return get_noc_addr(get_common_arg_val<uint32_t>(4 + 2 * K + 2 * l), get_common_arg_val<uint32_t>(5 + 2 * K + 2 * l),
                            addr);
    };

    // tile t's packed chunks are the 64 T bytes at 64 T t (face 0's T rows, then face 1's): this packer's are one run,
    // assembled in the staging CB (64 T bytes per tile) straight from the source where the alignment allows (face 0;
    // face 1 for even T), face 1 of odd T through the staging tail + an L1 copy
    constexpr uint32_t R0 = (32 * T + 63) / 64 * 64;  // (a DRAM read: 64 B-aligned like its source)
    constexpr uint32_t RUN = 64 * T * TP;
    cb_reserve_back(cb_stage, 1);
    const uint32_t asm_ = get_write_ptr(cb_stage), tail = asm_ + RUN;
    const uint32_t tx = get_write_ptr(cb_tx);
    for (uint32_t t = t0; t < t1; ++t) {
        const uint64_t s0 = at.get_noc_addr(t);
        const uint32_t d = asm_ + (t - t0) * 64 * T;
        noc_async_read(s0, d, R0);
        if constexpr (T % 2 == 0) {
            noc_async_read(s0 + 512, d + 32 * T, 32 * T);
        } else {
            noc_async_read(s0 + 512, tail + (t - t0) * R0, R0);
        }
    }
    noc_async_read_barrier();
    if constexpr (T % 2 == 1) {
        for (uint32_t t = t0; t < t1; ++t) {
            noc_async_read(get_noc_addr(tail + (t - t0) * R0), asm_ + (t - t0) * 64 * T + 32 * T, 32 * T);
        }
        noc_async_read_barrier();
    }
    uint32_t b = t0 * 64 * T, n = (t1 - t0) * 64 * T, src = asm_;
    while (n > 0) {  // (the run may cross a shard tile)
        const uint32_t k = b / 2048, o = b % 2048, m = 2048 - o < n ? 2048 - o : n;
        noc_async_write(src, shard_noc(k, slot + o), m);
        noc_async_write(src, send_noc(k / KPL, tx + (k % KPL) * 2048 + o), m);
        b += m;
        src += m;
        n -= m;
    }
    noc_async_write_barrier();
    const uint32_t k0 = (2 * t0 * T * 32) / 2048, k1 = (2 * t1 * T * 32 - 1) / 2048;
    for (uint32_t k = k0; k <= k1; ++k) {
        noc_semaphore_inc(shard_noc(k, get_semaphore(1)), 1);
    }
    for (uint32_t l = k0 / KPL; l <= k1 / KPL; ++l) {
        noc_semaphore_inc(send_noc(l, get_semaphore(0)), 1);
    }
    noc_async_atomic_barrier();
}
