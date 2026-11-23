// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Verify rows of the short convs (tt/bconv.py SEQ): G groups of R consecutive rows of one sequence. The fp32 state
// tiles as read hold the committed state of group g at row g R + a[g] (the previous pass's accepted row). Rewritten
// here, in L1, so that row g R + r of state j = z[r + j], z = [committed s0, s1, s2, the group's x rows]: the per-row
// conv step is then the sequential one, and its rotated states leave row g R + r = the state after row r.

#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// element offset of (row r, half h) in a 32 x 32 tile (faces of 16 x 16)
inline uint32_t seq_off(uint32_t r, uint32_t h) { return ((r >> 4) * 2 + h) * 256 + (r & 15) * 16; }

// s: the 3 state tiles (fp32, consecutive); x: the input tile (XE bytes per element: 2 bf16, 4 fp32); scratch: one
// fp32 tile (rows < 16 for the committed rows); acc: the G accepted rows.
template <uint32_t XE>
inline void seq_shift(uint32_t s, uint32_t x, uint32_t scratch, volatile tt_l1_ptr uint32_t* acc) {
    constexpr uint32_t R = SEQ_R, G = SEQ_G, F32 = 4096;
    static_assert(3 * G <= 16 && R * G <= 32, "verify rows");
    for (uint32_t g = 0; g < G; ++g) {
        const uint32_t ra = g * R + acc[g];
        for (uint32_t j = 0; j < 3; ++j) {
            for (uint32_t h = 0; h < 2; ++h) {
                auto src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s + j * F32 + seq_off(ra, h) * 4);
                auto dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + seq_off(g * 3 + j, h) * 4);
                for (uint32_t e = 0; e < 16; ++e) {
                    dst[e] = src[e];
                }
            }
        }
    }
    for (uint32_t g = 0; g < G; ++g) {
        for (uint32_t j = 0; j < 3; ++j) {
            for (uint32_t r = 0; r < R; ++r) {
                const uint32_t i = r + j;
                for (uint32_t h = 0; h < 2; ++h) {
                    auto dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s + j * F32 + seq_off(g * R + r, h) * 4);
                    if (i < 3) {
                        auto src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + seq_off(g * 3 + i, h) * 4);
                        for (uint32_t e = 0; e < 16; ++e) {
                            dst[e] = src[e];
                        }
                    } else if constexpr (XE == 4) {
                        auto src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(x + seq_off(g * R + i - 3, h) * 4);
                        for (uint32_t e = 0; e < 16; ++e) {
                            dst[e] = src[e];
                        }
                    } else {
                        auto src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(x + seq_off(g * R + i - 3, h) * 2);
                        for (uint32_t e = 0; e < 16; ++e) {
                            dst[e] = (uint32_t)src[e] << 16;
                        }
                    }
                }
            }
        }
    }
}

// the G accepted rows (uint32, one 128 B page) into scratch + 3072 (past the committed rows' faces)
template <typename A>
inline volatile tt_l1_ptr uint32_t* seq_acc(const A& acc, uint32_t scratch) {
    noc_async_read(acc.get_noc_addr(0), scratch + 3072, 128);
    noc_async_read_barrier();
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + 3072);
}
