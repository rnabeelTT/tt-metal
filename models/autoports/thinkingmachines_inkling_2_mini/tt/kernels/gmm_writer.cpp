// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Grouped expert matmul (tt/gmm.py), writer: the nb output tiles of each (group, row tile) -> C tiles
// (g mt + mi) Nt + n0 + j.
// GMM_WW: the writer also loads the weight blocks (one per run of groups of one expert; the reader's loop in
// gmm_reader.cpp), each as soon as a CB slot is free, so the next expert's weights stream while the reader feeds A.
// It never blocks on the weight CB: the output blocks drain while it waits for a slot and between block rows.
//
// GMM_NS: each row tile's NS output blocks go to C_0 .. C_{NS-1} (the sub-shards' products).
//
// runtime args: b, e0, e1 (this core's experts), ..., G (index 9), the first output column n0, (GMM_WW) s, c0
// common args: (GMM_WW) w, ws, wo, wkt, k0; c_0 .. c_{NS-1}; (GMM_W2, act: NS = 1) w2

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "gmm_bal.hpp"

#ifndef GMM_WW
#define GMM_WW 0
#endif
#ifndef GMM_NS
#define GMM_NS 1
#endif
#ifndef GMM_BAL
#define GMM_BAL 0
#endif
#ifndef GMM_GS
#define GMM_GS 0
#endif
#ifndef GMM_W2
#define GMM_W2 0
#endif

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t Nt = get_compile_time_arg_val(1);
    constexpr uint32_t nb = get_compile_time_arg_val(2);
    constexpr uint32_t mt = get_compile_time_arg_val(3);
    constexpr uint32_t cb_ids = get_compile_time_arg_val(4);
    constexpr uint32_t E = get_compile_time_arg_val(5);
    constexpr auto c_args = TensorAccessorArgs<6>();
    const uint32_t b = get_arg_val<uint32_t>(0);
    uint32_t e0 = get_arg_val<uint32_t>(1), e1 = get_arg_val<uint32_t>(2);  // (GMM_BAL: s, S)
    const uint32_t G = get_arg_val<uint32_t>(9), n0 = get_arg_val<uint32_t>(10);
#if GMM_RM
    constexpr uint32_t c_page = Nt * 64;
#else
    constexpr uint32_t c_page = 2048;
#endif
    cb_wait_front(cb_ids, 1);  // (the reader's copy of the group ids)
    volatile tt_l1_ptr uint16_t* ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(cb_ids));
#if GMM_BAL
    gmm_bal([&](uint32_t g) -> uint32_t { return ids[g]; }, G, E, e0, e1, GMM_BAL, e0, e1);
#endif
    uint32_t g_lo = 0, g_hi = G;
#if GMM_GS
    gmm_gs([&](uint32_t g) -> uint32_t { return ids[g]; }, G, E, e0, e1, g_lo, g_hi);
    e0 = 0;
    e1 = E;
#endif
    auto next_group = [&](uint32_t g) {  // this core's first group >= g, or g_hi (sorted; padding ids >= E >= e1)
        for (; g < g_hi && ids[g] < e1; ++g) {
            if (ids[g] >= e0) {
                return g;
            }
        }
        return g_hi;
    };
    uint32_t og = next_group(g_lo), omi = 0, os = 0;  // (the next output block: group, row tile, sub-shard)
    // writes the next output block (block: waits for it; else only if the compute has it ready); false when all done
    auto drain = [&](bool block) {
        if (og >= g_hi) {
            return false;
        }
        if (!block && !cb_pages_available_at_front(cb_out, nb)) {
            return true;
        }
        cb_wait_front(cb_out, nb);
        const uint32_t l = get_read_ptr(cb_out);
        const auto ac = TensorAccessor(c_args, get_common_arg_val<uint32_t>(5 + os), c_page);
#if GMM_RM
        // row-major C (pages = rows of Nt * 64 bytes): this block's 32 rows, nb * 64 bytes each at column b nb
        for (uint32_t r = 0; r < 32; ++r) {
            noc_async_write(l + r * nb * 64, ac.get_noc_addr((og * mt + omi) * 32 + r) + n0 * 64, nb * 64);
        }
#else
        for (uint32_t j = 0; j < nb; ++j) {
            noc_async_write(l + j * 2048, ac.get_noc_addr((og * mt + omi) * Nt + n0 + j), 2048);
        }
#endif
        noc_async_write_barrier();
        cb_pop_front(cb_out, nb);
        if (++os == GMM_NS) {
            os = 0;
            if (++omi == mt) {
                omi = 0;
                og = next_group(og + 1);
            }
        }
        return true;
    };
#if GMM_WW
    constexpr uint32_t w0 = c_args.next_compile_time_args_offset();
    constexpr uint32_t cb_w = get_compile_time_arg_val(w0);
    constexpr uint32_t Kt = get_compile_time_arg_val(w0 + 1);
    constexpr uint32_t Nw = get_compile_time_arg_val(w0 + 2);  // (w's column tiles)
    constexpr uint32_t w_page = get_compile_time_arg_val(w0 + 3);
    constexpr uint32_t Wt = get_compile_time_arg_val(w0 + 4);  // (GMM_ACT: the up columns' offset)
    constexpr auto w_args = TensorAccessorArgs<w0 + 5>();
#if GMM_ACT
    constexpr uint32_t WB = 2 * nb;  // (nb is h here: the block holds the h gate and the h up columns)
#else
    constexpr uint32_t WB = nb;
#endif
    constexpr uint32_t KT = Kt * GMM_NS;
    const auto aw = TensorAccessor(w_args, get_common_arg_val<uint32_t>(0), w_page);
    const uint32_t ws = get_common_arg_val<uint32_t>(1), wo = get_common_arg_val<uint32_t>(2) + get_arg_val<uint32_t>(11);
    const uint32_t wkt = get_common_arg_val<uint32_t>(3), k0 = get_common_arg_val<uint32_t>(4);
    const uint32_t c0 = get_arg_val<uint32_t>(12);
#if GMM_W2
    const auto aw2 = TensorAccessor(w_args, get_common_arg_val<uint32_t>(6), w_page);
#endif
    uint32_t prev = 0xFFFFFFFF;
    for (uint32_t g = next_group(g_lo); g < g_hi; g = next_group(g + 1)) {
        const uint32_t e = ids[g];
        if (e == prev) {
            continue;
        }
        while (!cb_pages_reservable_at_back(cb_w, KT * WB)) {
            drain(false);
        }
        cb_reserve_back(cb_w, KT * WB);
        const uint32_t wl = get_write_ptr(cb_w);
        for (uint32_t st = 0; st < KT; ++st) {
            const uint32_t s = st / Kt, kt = st % Kt;
            for (uint32_t j = 0; j < WB; ++j) {
#if GMM_W2
                bool second;
                const uint32_t col = gmm_w2_col<Wt, GMM_W2, WB / 2>(c0, j, second);
                const uint32_t row = (e * ws + wo + s) * wkt + k0 + kt;
                noc_async_read(second ? aw2.get_noc_addr(row * 2 * GMM_W2 + col) : aw.get_noc_addr(row * Nw + col),
                               wl + (st * WB + j) * w_page, w_page);
#else
#if GMM_ACT
                const uint32_t col = j < WB / 2 ? c0 + j : Wt + c0 + j - WB / 2;
#else
                const uint32_t col = c0 + j;
#endif
                noc_async_read(aw.get_noc_addr(((e * ws + wo + s) * wkt + k0 + kt) * Nw + col),
                               wl + (st * WB + j) * w_page, w_page);
#endif
            }
            if ((st & 7) == 7) {
                drain(false);
            }
        }
        while (!ncrisc_noc_reads_flushed(noc_index)) {
            drain(false);
        }
        noc_async_read_barrier();
        cb_push_back(cb_w, KT * WB);
        prev = e;
    }
#endif
    while (drain(true)) {
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
