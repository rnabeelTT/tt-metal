// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Verify rows of the short convs (tt/bconv.py SEQ): G groups of R consecutive rows of one sequence. The fp32 state
// tiles as read hold the committed state of group g at row g R + a[g] (the previous pass's accepted row). Rewritten
// here, in L1, so that row g R + r of state j = z[r + j], z = [committed s0, s1, s2, the group's x rows]: the per-row
// conv step is then the sequential one, and its rotated states leave row g R + r = the state after row r.
//
// z is staged as fp32 rows in a scratch tile (row g 8 + i = z[i], i < R + 2: rows past 15 sit in faces 2 / 3), then
// every state row is one local NOC copy per half row (64 B), all in flight at once.

#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// element offset of (row r, half h) in a 32 x 32 tile (faces of 16 x 16)
inline uint32_t seq_off(uint32_t r, uint32_t h) { return ((r >> 4) * 2 + h) * 256 + (r & 15) * 16; }

// s: the 3 state tiles (fp32, consecutive); x: the input tile (XE bytes per element: 2 bf16, 4 fp32); scratch: one
// fp32 tile (z rows); acc: the G accepted rows.
template <uint32_t XE>
inline void seq_shift(uint32_t s, uint32_t x, uint32_t scratch, volatile tt_l1_ptr uint32_t* acc) {
    constexpr uint32_t R = SEQ_R, G = SEQ_G, F32 = 4096;
    static_assert(R + 2 <= 8 && G <= 4 && R * G <= 32, "verify rows");
    // z rows: the committed state (3 rows: local copies) and the first R - 1 inputs (converted to fp32 on this core)
    for (uint32_t g = 0; g < G; ++g) {
        const uint32_t ra = g * R + acc[g];
        for (uint32_t j = 0; j < 3; ++j) {
            for (uint32_t h = 0; h < 2; ++h) {
                noc_async_read(get_noc_addr(s + j * F32 + seq_off(ra, h) * 4), scratch + seq_off(g * 8 + j, h) * 4, 64);
            }
        }
        for (uint32_t i = 0; i + 1 < R; ++i) {
            for (uint32_t h = 0; h < 2; ++h) {
                auto dst = reinterpret_cast<tt_l1_ptr uint32_t*>(scratch + seq_off(g * 8 + 3 + i, h) * 4);
                if constexpr (XE == 4) {
                    auto src = reinterpret_cast<tt_l1_ptr uint32_t*>(x + seq_off(g * R + i, h) * 4);
                    for (uint32_t e = 0; e < 16; ++e) {
                        dst[e] = src[e];
                    }
                } else {
                    auto src = reinterpret_cast<tt_l1_ptr uint16_t*>(x + seq_off(g * R + i, h) * 2);
                    for (uint32_t e = 0; e < 16; ++e) {
                        dst[e] = (uint32_t)src[e] << 16;
                    }
                }
            }
        }
    }
    noc_async_read_barrier();
    // state j row g R + r = z[r + j]
    for (uint32_t g = 0; g < G; ++g) {
        for (uint32_t j = 0; j < 3; ++j) {
            for (uint32_t r = 0; r < R; ++r) {
                for (uint32_t h = 0; h < 2; ++h) {
                    noc_async_read(get_noc_addr(scratch + seq_off(g * 8 + r + j, h) * 4),
                                   s + j * F32 + seq_off(g * R + r, h) * 4, 64);
                }
            }
        }
    }
    noc_async_read_barrier();
}

// the G accepted rows (uint32, one 128 B page): at scratch + 3968 (face 3, rows 30 / 31; the z rows end at row
// 8 (G - 1) + R + 1 <= 29)
template <typename A>
inline volatile tt_l1_ptr uint32_t* seq_acc(const A& acc, uint32_t scratch) {
    noc_async_read(acc.get_noc_addr(0), scratch + 3968, 128);
    noc_async_read_barrier();
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + 3968);
}
