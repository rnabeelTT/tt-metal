// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// DSpark Markov step, activations + argmax (BRISC; tt/drafter.py MarkovStep). Row 0 of the kt - 1 bf16 latent tiles
// into cb_x plus a ones tile (x[0, 256] = 1: the fold bias row), this core's base logits (row `row` of the compact
// [1, 8, NT, 32] FP32 tensor) as one FP32 tile per column (row 0 live) into cb_base; then row 0 of each output tile is
// scanned for this core's max (integer order of the FP32 bits; ascending vocab ids, strict >: lowest id on ties;
// ids >= V skipped). Every core sends (key, id) to the reducer's cb_cand slot and bumps its semaphore; the reducer
// picks the global max (ties: lowest id) and writes the token to `prev` (uint32 [1, 1] ROW_MAJOR).
//
// runtime args: [x_addr, base_addr, prev_addr, col0, ncols, slot]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

static inline uint32_t fkey(uint32_t u) { return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t base_addr = get_arg_val<uint32_t>(1);
    const uint32_t prev_addr = get_arg_val<uint32_t>(2);
    const uint32_t col0 = get_arg_val<uint32_t>(3);
    const uint32_t ncols = get_arg_val<uint32_t>(4);
    const uint32_t slot = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t cb_base = get_compile_time_arg_val(2);
    constexpr uint32_t cb_bseg = get_compile_time_arg_val(3);
    constexpr uint32_t cb_cand = get_compile_time_arg_val(4);
    constexpr uint32_t kt = get_compile_time_arg_val(5);
    constexpr uint32_t V = get_compile_time_arg_val(6);
    constexpr uint32_t NT = get_compile_time_arg_val(7);
    constexpr uint32_t row = get_compile_time_arg_val(8);
    constexpr uint32_t ncores = get_compile_time_arg_val(9);
    constexpr uint32_t red_x = get_compile_time_arg_val(10);
    constexpr uint32_t red_y = get_compile_time_arg_val(11);
    constexpr auto x_args = TensorAccessorArgs<12>();
    constexpr auto b_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();

    const uint32_t sem = get_semaphore(0);
    const uint32_t cand = get_write_ptr(cb_cand);
    uint32_t best_key = 0, best_id = 0xFFFFFFFFu;

    if (ncols > 0) {
        cb_reserve_back(cb_x, kt);
        const uint32_t xl1 = get_write_ptr(cb_x);
        const auto s_x = TensorAccessor(x_args, x_addr, 2048);
        for (uint32_t k = 0; k + 1 < kt; ++k) {
            const uint64_t src = s_x.get_noc_addr(k);
            noc_async_read(src, xl1 + k * 2048, 32);
            noc_async_read(src + 512, xl1 + k * 2048 + 512, 32);
        }
        volatile tt_l1_ptr uint32_t* ones = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xl1 + (kt - 1) * 2048);
        for (uint32_t i = 0; i < 8; ++i) {
            ones[i] = 0;
            ones[128 + i] = 0;
        }
        ones[0] = 0x3F80u;  // bf16 1.0 at [0, 0]
        const uint32_t bseg = get_write_ptr(cb_bseg);
        const auto s_b = TensorAccessor(b_args, base_addr, 128);
        for (uint32_t i = 0; i < ncols; ++i) {
            noc_async_read(s_b.get_noc_addr(row * NT + col0 + i), bseg + i * 128, 128);
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, kt);

        for (uint32_t i = 0; i < ncols; ++i) {
            cb_reserve_back(cb_base, 1);
            volatile tt_l1_ptr uint32_t* bt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_base));
            volatile tt_l1_ptr uint32_t* bs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bseg + i * 128);
            for (uint32_t j = 0; j < 16; ++j) {
                bt[j] = bs[j];
                bt[256 + j] = bs[16 + j];
            }
            cb_push_back(cb_base, 1);

            cb_wait_front(cb_out, 1);
            volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_out));
            const uint32_t v0 = (col0 + i) * 32;
            for (uint32_t j = 0; j < 32; ++j) {
                const uint32_t v = v0 + j;
                if (v >= V) {
                    break;
                }
                const uint32_t key = fkey(t[j < 16 ? j : 240 + j]);
                if (key > best_key) {
                    best_key = key;
                    best_id = v;
                }
            }
            cb_pop_front(cb_out, 1);
        }
    }
    // candidate -> the reducer
    volatile tt_l1_ptr uint32_t* mine = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cand + ncores * 16);
    mine[0] = best_key;
    mine[1] = best_id;
    noc_async_write(cand + ncores * 16, get_noc_addr(red_x, red_y, cand + slot * 16), 16);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(red_x, red_y, sem), 1);
    noc_async_atomic_barrier();
    if (slot != 0) {
        return;
    }
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem), ncores);
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem), 0);
    volatile tt_l1_ptr uint32_t* c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cand);
    uint32_t bk = 0, bi = 0xFFFFFFFFu;
    for (uint32_t s = 0; s < ncores; ++s) {
        const uint32_t k = c[s * 4], id = c[s * 4 + 1];
        if (k > bk || (k == bk && id < bi)) {
            bk = k;
            bi = id;
        }
    }
    mine[0] = bi;
    const auto s_p = TensorAccessor(p_args, prev_addr, 4);
    noc_async_write(cand + ncores * 16, s_p.get_noc_addr(0), 4);
    noc_async_write_barrier();
}
