// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The cross-core reductions' gather layout (tt/post.py red_sel): core c's partial row -> tile c / (4 K), row
// 8 ((c % 4 K) / K) + c % K. The FPU sums a matmul's inner dim in 8-row groups at reduced precision (the group sums
// then accumulate in fp32), so at most K partials share a group; the unused rows are zeroed (0 x NaN is NaN).
// TR > 1 (RED_PACK, K TR <= 8): core c's TR rows at group rows (c % K) TR .. + TR - 1, reduced by one root whose
// selection tile row t picks rows (c % K) TR + t: each row's group still adds the same K partials (the rest x 0).

#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

inline uint32_t red_row_off(uint32_t i) { return (i < 16 ? 0u : 2048u) + (i % 16) * 64; }

// byte offset of core c's row in a part's gather tiles
template <uint32_t K, uint32_t TR = 1>
inline uint32_t red_off(uint32_t c) {
    return (c / (4 * K)) * 4096 + red_row_off(8 * ((c % (4 * K)) / K) + (c % K) * TR);
}

#ifndef RED_WS
#define RED_WS 0
#endif

// zero the unused rows of NQ gather tiles at base (local L1) from the zero block z (>= 512 B). RED_WS: stateful
// one-packet writes (the destination core and size programmed once)
template <uint32_t K, uint32_t C, uint32_t NQ, uint32_t TR = 1>
inline void red_zero(uint32_t z, uint32_t base) {
    uint32_t cur = 0;
    for (uint32_t k = 0; k < NQ; ++k) {
        for (uint32_t grp = 0; grp < 4; ++grp) {
            const uint32_t first = k * 4 * K + grp * K;
            const uint32_t used = (first >= C ? 0 : (C - first < K ? C - first : K)) * TR;
            if (used < 8) {
                const uint32_t o = base + k * 4096 + red_row_off(8 * grp + used), n = (8 - used) * 64;
                if constexpr (RED_WS) {
                    if (n != cur) {
                        noc_async_write_one_packet_set_state(get_noc_addr(o), n);
                        cur = n;
                    }
                    noc_async_write_one_packet_with_state(z, o);
                    noc_async_write_one_packet_with_state(z, o + 1024);
                } else {
                    noc_async_write(z, get_noc_addr(o), n);
                    noc_async_write(z, get_noc_addr(o + 1024), n);
                }
            }
        }
    }
}

// a sender's row (src: row t of part p0's CB tile; parts FT apart, the second face FO on) of parts p0 .. p1 - 1 ->
// the root at dst (NOC address of its gather row d; part p at d + p NQ tiles)
template <uint32_t NQ, uint32_t FT, uint32_t FO, uint32_t B = 64>
inline void red_send(uint32_t src, uint64_t dst, uint32_t d, uint32_t p0, uint32_t p1) {
    if constexpr (RED_WS) {
        noc_async_write_one_packet_set_state(dst, B);
        for (uint32_t p = p0; p < p1; ++p) {
            noc_async_write_one_packet_with_state(src + p * FT, d + p * NQ * 4096);
            noc_async_write_one_packet_with_state(src + p * FT + FO, d + p * NQ * 4096 + 1024);
        }
    } else {
        for (uint32_t p = p0; p < p1; ++p) {
            noc_async_write(src + p * FT, dst + p * NQ * 4096, B);
            noc_async_write(src + p * FT + FO, dst + p * NQ * 4096 + 1024, B);
        }
    }
}
