// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The RP_ROWS (R <= 8) live rows of a TILE tensor [1, 1, R, 32 T] <-> dense tiles [1, R, 32, 32 TD] (tt/sampling.py,
// I2M_TOPK_DENSE): row r of tile t is row t % 32 of dense tile r TD + t / 32 (its two 16-column face halves to the same
// face halves of that row). Elementwise ops on the dense copy touch R TD tiles instead of T (the 32 - R padding rows of
// every tile are never read).
// RP_PACK: fp32 source tiles -> fp32 dense tiles; else bf16 dense tiles -> rows 0 .. R - 1 of bf16 tiles.
//
// runtime args: src addr, dst addr, first tile t, tiles, staging area (0 / 1) [RP_MAX: worker index, out addr]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// RP_MAX (with RP_PACK): also row r's bf16(max) as fp32 in column 0 of all 32 rows of tile r of the R-tile output
// [1, R, 32, 1] (the merge as tt/kernels/rowmax.cpp: every core's max keys to core 0's slot area, then core 0 writes)
static inline uint32_t key_of(uint32_t u) { return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }
static inline uint32_t val_of(uint32_t k) { return (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k; }

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t BATCH = get_compile_time_arg_val(1);
    constexpr auto s_args = TensorAccessorArgs<3>();
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
    constexpr uint32_t R = RP_ROWS;
    constexpr uint32_t TD = get_compile_time_arg_val(2);  // dense tiles per row block
    const auto s = TensorAccessor(s_args, s_addr, TB);
    const auto d = TensorAccessor(d_args, d_addr, TB);
    // NS workers per core (BRISC / NCRISC on NOC 0 / 1), each its own staging area: BATCH x 2 faces x R rows (64 B each)
    constexpr uint32_t NS = get_compile_time_arg_val(d_args.next_compile_time_args_offset());
    const uint32_t stage = get_write_ptr(cb) + get_arg_val<uint32_t>(4) * BATCH * R * 2 * 64;
#if RP_MAX
    constexpr uint32_t NC = get_compile_time_arg_val(d_args.next_compile_time_args_offset() + 1);  // workers
    constexpr uint32_t MX = get_compile_time_arg_val(d_args.next_compile_time_args_offset() + 2);
    constexpr uint32_t MY = get_compile_time_arg_val(d_args.next_compile_time_args_offset() + 3);
    constexpr auto o_args = TensorAccessorArgs<d_args.next_compile_time_args_offset() + 4>();
    const uint32_t me = get_arg_val<uint32_t>(5), o_addr = get_arg_val<uint32_t>(6);
    // after the NS staging areas: NC x 32 B keys (worker 0's merge; each worker stages its own at its slot), then the
    // output tiles
    const uint32_t slots = get_write_ptr(cb) + NS * BATCH * R * 2 * 64;
    uint32_t key[R];
    for (uint32_t r = 0; r < R; ++r) {
        key[r] = 0;
    }
#endif
    auto dense_off = [](uint32_t r, uint32_t f) {  // row r, face half f of a dense tile
        return ((r >> 4) * 2 + f) * FB + (r & 15) * RB;
    };
    for (uint32_t b = 0; b < nt; b += BATCH) {
        const uint32_t n = nt - b < BATCH ? nt - b : BATCH;
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t t = t0 + b + i;
            for (uint32_t f = 0; f < 2; ++f) {
                const uint32_t st = stage + (i * 2 + f) * R * 64;  // (rows 0 .. R - 1 of face f: contiguous)
#if RP_PACK
                noc_async_read(s.get_noc_addr(t) + f * FB, st, R * RB);
#else
                // (the 64 B-aligned row pair holding the row: the dense copy may be in DRAM; compacted below)
                const uint32_t off = dense_off(t % 32, f);
                for (uint32_t r = 0; r < R; ++r) {
                    noc_async_read(s.get_noc_addr(r * TD + t / 32) + (off & ~63u), st + r * 64, 64);
                }
#endif
            }
        }
        noc_async_read_barrier();
#if !RP_PACK
        invalidate_l1_cache();
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t o = (dense_off((t0 + b + i) % 32, 0) & 63u) / 4;  // (the same for both faces)
            for (uint32_t f = 0; f < 2; ++f) {
                volatile tt_l1_ptr uint32_t* q = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage + (i * 2 + f) * R * 64);
                for (uint32_t r = 0; r < R; ++r) {  // row r: word 16 r + o -> word 8 r (increasing r: no overlap)
#pragma GCC unroll 8
                    for (uint32_t w = 0; w < 8; ++w) {
                        q[r * 8 + w] = q[r * 16 + o + w];
                    }
                }
            }
        }
#endif
#if RP_MAX
        invalidate_l1_cache();
        {
            const uint32_t* p = reinterpret_cast<const uint32_t*>(stage);
            for (uint32_t i = 0; i < n; ++i) {
                for (uint32_t r = 0; r < R; ++r) {
                    uint32_t k = key[r];
                    for (uint32_t f = 0; f < 2; ++f) {
                        const uint32_t* q = p + ((i * 2 + f) * R + r) * 16;
#pragma GCC unroll 16
                        for (uint32_t c = 0; c < 16; ++c) {
                            const uint32_t kk = key_of(q[c]);
                            k = kk > k ? kk : k;
                        }
                    }
                    key[r] = k;
                }
            }
        }
#endif
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t t = t0 + b + i;
            for (uint32_t f = 0; f < 2; ++f) {
                const uint32_t st = stage + (i * 2 + f) * R * 64;
#if RP_PACK
                for (uint32_t r = 0; r < R; ++r) {
                    noc_async_write(st + r * RB, d.get_noc_addr(r * TD + t / 32) + dense_off(t % 32, f), RB);
                }
#else
                noc_async_write(st, d.get_noc_addr(t) + f * FB, R * RB);
#endif
            }
        }
        noc_async_write_barrier();
    }
#if RP_MAX
    volatile tt_l1_ptr uint32_t* sl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slots);
    if (me != 0) {
        for (uint32_t r = 0; r < R; ++r) {
            sl[me * 8 + r] = key[r];
        }
        noc_async_write(slots + me * 32, get_noc_addr(MX, MY, slots + me * 32), 32);
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
        for (uint32_t r = 0; r < R; ++r) {
            const uint32_t kk = sl[c * 8 + r];
            key[r] = kk > key[r] ? kk : key[r];
        }
    }
    const uint32_t ot = slots + NC * 32;
    volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ot);
    const auto so = TensorAccessor(o_args, o_addr, 4096);
    for (uint32_t r = 0; r < R; ++r) {  // (R output tiles from R staging tiles)
        volatile tt_l1_ptr uint32_t* ob = o + r * 1024;
        for (uint32_t i = 0; i < 1024; ++i) {
            ob[i] = 0;
        }
        const uint32_t u = val_of(key[r]), v = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
        for (uint32_t q = 0; q < 32; ++q) {
            ob[(q >> 4) * 512 + (q & 15) * 16] = v;
        }
        noc_async_write(ot + r * 4096, so.get_noc_addr(r), 4096);
    }
    noc_async_write_barrier();
#endif
}
