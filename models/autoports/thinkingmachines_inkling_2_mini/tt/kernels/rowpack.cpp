// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// One live row of a TILE tensor [1, 1, 1, 32 T] <-> dense tiles (tt/sampling.py, I2M_TOPK_DENSE): row 0 of tile t is
// row t % 32 of dense tile t / 32 (its two 16-column face halves to the same face halves of that row). Elementwise ops
// on the dense copy touch T / 32 tiles instead of T (the 31 padding rows of every tile are never read).
// RP_PACK: fp32 source tiles -> fp32 dense tiles; else bf16 dense tiles -> row 0 of bf16 tiles.
//
// runtime args: src addr, dst addr, first tile t, tiles

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// RP_MAX (with RP_PACK): also the row's bf16(max) as fp32 in column 0 of all 32 rows of the one-tile output (the merge
// as tt/kernels/rowmax.cpp: every core's max key to core 0's slot area, then core 0 writes the tile)
static inline uint32_t key_of(uint32_t u) { return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }
static inline uint32_t val_of(uint32_t k) { return (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k; }

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t BATCH = get_compile_time_arg_val(1);
    constexpr auto s_args = TensorAccessorArgs<2>();
    constexpr auto d_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    const uint32_t s_addr = get_arg_val<uint32_t>(0), d_addr = get_arg_val<uint32_t>(1);
    const uint32_t t0 = get_arg_val<uint32_t>(2), nt = get_arg_val<uint32_t>(3);
#if RP_PACK
    constexpr uint32_t E = 4, TB = 4096;
#else
    constexpr uint32_t E = 2, TB = 2048;
#endif
    constexpr uint32_t RB = 16 * E;          // one face row
    constexpr uint32_t FB = 256 * E;         // one face
    const auto s = TensorAccessor(s_args, s_addr, TB);
    const auto d = TensorAccessor(d_args, d_addr, TB);
    const uint32_t stage = get_write_ptr(cb);  // BATCH x 2 face rows, each in its own 64 B slot
#if RP_MAX
    constexpr uint32_t NC = get_compile_time_arg_val(d_args.next_compile_time_args_offset());
    constexpr uint32_t MX = get_compile_time_arg_val(d_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t MY = get_compile_time_arg_val(d_args.next_compile_time_args_offset() + 2);
    constexpr auto o_args = TensorAccessorArgs<d_args.next_compile_time_args_offset() + 3>();
    const uint32_t me = get_arg_val<uint32_t>(4), o_addr = get_arg_val<uint32_t>(5);
    const uint32_t slots = stage + BATCH * 2 * 64;  // NC x 16 B keys (core 0), then the output tile
    uint32_t key = 0;
#endif
    auto dense_off = [](uint32_t r, uint32_t f) {  // row r, face half f of a dense tile
        return ((r >> 4) * 2 + f) * FB + (r & 15) * RB;
    };
    for (uint32_t b = 0; b < nt; b += BATCH) {
        const uint32_t n = nt - b < BATCH ? nt - b : BATCH;
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t t = t0 + b + i;
            for (uint32_t f = 0; f < 2; ++f) {
#if RP_PACK
                noc_async_read(s.get_noc_addr(t) + f * FB, stage + (2 * i + f) * 64, RB);
#else
                noc_async_read(s.get_noc_addr(t / 32) + dense_off(t % 32, f), stage + (2 * i + f) * 64 + 0, RB);
#endif
            }
        }
        noc_async_read_barrier();
#if RP_MAX
        invalidate_l1_cache();
        {
            const uint32_t* p = reinterpret_cast<const uint32_t*>(stage);
            for (uint32_t j = 0; j < 2 * n; ++j) {
#pragma GCC unroll 16
                for (uint32_t c = 0; c < 16; ++c) {
                    const uint32_t kk = key_of(p[j * 16 + c]);
                    key = kk > key ? kk : key;
                }
            }
        }
#endif
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t t = t0 + b + i;
            for (uint32_t f = 0; f < 2; ++f) {
#if RP_PACK
                noc_async_write(stage + (2 * i + f) * 64, d.get_noc_addr(t / 32) + dense_off(t % 32, f), RB);
#else
                noc_async_write(stage + (2 * i + f) * 64, d.get_noc_addr(t) + f * FB, RB);
#endif
            }
        }
        noc_async_write_barrier();
    }
#if RP_MAX
    volatile tt_l1_ptr uint32_t* sl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slots);
    if (me != 0) {
        sl[0] = key;
        noc_async_write(slots, get_noc_addr(MX, MY, slots + me * 16), 16);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(MX, MY, get_semaphore(0)), 1);
        noc_async_atomic_barrier();
        return;
    }
    if constexpr (NC > 1) {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0)), NC - 1);
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0)), 0);
        invalidate_l1_cache();
    }
    for (uint32_t c = 1; c < NC; ++c) {
        const uint32_t kk = sl[c * 4];
        key = kk > key ? kk : key;
    }
    const uint32_t ot = slots + NC * 16;
    volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ot);
    for (uint32_t i = 0; i < 1024; ++i) {
        o[i] = 0;
    }
    const uint32_t u = val_of(key), v = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
    for (uint32_t r = 0; r < 32; ++r) {
        o[(r >> 4) * 512 + (r & 15) * 16] = v;
    }
    const auto so = TensorAccessor(o_args, o_addr, 4096);
    noc_async_write(ot, so.get_noc_addr(0), 4096);
    noc_async_write_barrier();
#endif
}
