// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The decode sampling's shift (tt/sampling.py DeviceTopK.device, I2M_TOPK_MAX): per live row of fp32 TILE logits
// [1, 1, rows, Vp], bf16(max_j x_j) as fp32 into column 0 of a one-tile fp32 output. bf16 rounding is monotone, so this
// equals the first top-k pass's best value (the max over the bf16-typecast row). Each core scans a range of tiles
// (only the live rows' face-0 / face-1 runs: ROWS x 64 B each), the first core merges every core's maxima.
//
// runtime args: logits addr, out addr, first tile, tiles, core index; compile: cb, ROWS, cores, merge noc x / y, sem

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

static inline uint32_t key_of(uint32_t u) { return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }
static inline uint32_t val_of(uint32_t k) { return (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k; }

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t ROWS = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t MX = get_compile_time_arg_val(3);
    constexpr uint32_t MY = get_compile_time_arg_val(4);
    constexpr uint32_t SEM = get_compile_time_arg_val(5);
    constexpr uint32_t BATCH = get_compile_time_arg_val(6);  // tiles per read round
    constexpr uint32_t FILL = get_compile_time_arg_val(7);   // (ROWS 1) the value in all 32 rows of the output
    constexpr auto l_args = TensorAccessorArgs<8>();
    constexpr auto o_args = TensorAccessorArgs<l_args.next_compile_time_args_offset()>();
    const uint32_t l_addr = get_arg_val<uint32_t>(0), o_addr = get_arg_val<uint32_t>(1);
    const uint32_t t0 = get_arg_val<uint32_t>(2), nt = get_arg_val<uint32_t>(3), me = get_arg_val<uint32_t>(4);
    const auto s_l = TensorAccessor(l_args, l_addr, 4096);
    constexpr uint32_t RUN = ROWS * 64;  // one face's live rows (16 fp32 each)
    constexpr uint32_t SW = (ROWS + 3) / 4 * 4;  // words per core slot (16 B-aligned NOC writes)
    const uint32_t base = get_write_ptr(cb);
    const uint32_t slots = base;                       // NC x ROWS keys (the merge core's gather)
    const uint32_t stage = base + ((NC * SW * 4 + 63) & ~63u);  // BATCH x 2 runs
    uint32_t key[ROWS];
    for (uint32_t r = 0; r < ROWS; ++r) {
        key[r] = 0;
    }
    for (uint32_t b = 0; b < nt; b += BATCH) {
        const uint32_t n = nt - b < BATCH ? nt - b : BATCH;
        for (uint32_t i = 0; i < n; ++i) {
            const uint64_t src = s_l.get_noc_addr(t0 + b + i);
            noc_async_read(src, stage + i * 2 * RUN, RUN);
            noc_async_read(src + 1024, stage + i * 2 * RUN + RUN, RUN);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();  // (Blackhole: the RISC's L1 data cache holds the previous batch's words)
        const uint32_t* p = reinterpret_cast<const uint32_t*>(stage);  // (plain loads: the cache was invalidated)
        for (uint32_t r = 0; r < ROWS; ++r) {
            uint32_t k = key[r];
            for (uint32_t j = 0; j < 2 * n; ++j) {
                const uint32_t* q = p + j * ROWS * 16 + r * 16;
#pragma GCC unroll 16
                for (uint32_t c = 0; c < 16; ++c) {
                    const uint32_t kk = key_of(q[c]);
                    k = kk > k ? kk : k;
                }
            }
            key[r] = k;
        }
    }
    volatile tt_l1_ptr uint32_t* sl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slots);
    if (me != 0) {
        for (uint32_t r = 0; r < ROWS; ++r) {
            sl[r] = key[r];
        }
        noc_async_write(slots, get_noc_addr(MX, MY, slots + me * SW * 4), SW * 4);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(MX, MY, get_semaphore(SEM)), 1);
        noc_async_atomic_barrier();
        return;
    }
    if constexpr (NC > 1) {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM)), NC - 1);
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM)), 0);
        invalidate_l1_cache();  // (the other cores' keys landed by NOC)
    }
    for (uint32_t c = 1; c < NC; ++c) {
        for (uint32_t r = 0; r < ROWS; ++r) {
            const uint32_t kk = sl[c * SW + r];
            key[r] = kk > key[r] ? kk : key[r];
        }
    }
    // the output tile: zeros, row r column 0 = bf16(max) (round to nearest even) as fp32
    volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage);
    for (uint32_t i = 0; i < 1024; ++i) {
        o[i] = 0;
    }
    for (uint32_t r = 0; r < (FILL ? 32 : ROWS); ++r) {
        const uint32_t u = val_of(key[FILL ? 0 : r]);
        o[(r >> 4) * 512 + (r & 15) * 16] = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
    }
    const auto s_o = TensorAccessor(o_args, o_addr, 4096);
    noc_async_write(stage, s_o.get_noc_addr(0), 4096);
    noc_async_write_barrier();
}
