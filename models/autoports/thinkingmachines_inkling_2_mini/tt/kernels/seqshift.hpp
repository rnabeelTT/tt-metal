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

// the same in a TH x 32 tile (TH < 32: two faces of TH x 16)
template <uint32_t TH>
inline uint32_t seq_off_h(uint32_t r, uint32_t h) {
    if constexpr (TH == 32) {
        return seq_off(r, h);
    } else {
        return h * TH * 16 + r * 16;
    }
}

// s: the 3 state tiles (fp32, consecutive); x: the input tile (XE bytes per element: 2 bf16, 4 fp32); scratch: one
// fp32 tile (z rows); acc: the G accepted rows. TH: the rows of the state / input tiles (8 / 16: TH x 32 CB tiles).
template <uint32_t XE, uint32_t TH = 32>
inline void seq_shift(uint32_t s, uint32_t x, uint32_t scratch, volatile tt_l1_ptr uint32_t* acc) {
    constexpr uint32_t R = SEQ_R, G = SEQ_G, F32 = TH * 128;
    static_assert(R + 2 <= 8 && G <= 4 && R * G <= TH, "verify rows");
    // z rows: the committed state (3 rows: local copies) and the first R - 1 inputs (converted to fp32 on this core)
    for (uint32_t g = 0; g < G; ++g) {
        const uint32_t ra = g * R + acc[g];
        for (uint32_t j = 0; j < 3; ++j) {
            for (uint32_t h = 0; h < 2; ++h) {
                noc_async_read(get_noc_addr(s + j * F32 + seq_off_h<TH>(ra, h) * 4), scratch + seq_off(g * 8 + j, h) * 4, 64);
            }
        }
        for (uint32_t i = 0; i + 1 < R; ++i) {
            for (uint32_t h = 0; h < 2; ++h) {
                auto dst = reinterpret_cast<tt_l1_ptr uint32_t*>(scratch + seq_off(g * 8 + 3 + i, h) * 4);
                if constexpr (XE == 4) {
                    auto src = reinterpret_cast<tt_l1_ptr uint32_t*>(x + seq_off_h<TH>(g * R + i, h) * 4);
                    for (uint32_t e = 0; e < 16; ++e) {
                        dst[e] = src[e];
                    }
                } else {
                    auto src = reinterpret_cast<tt_l1_ptr uint16_t*>(x + seq_off_h<TH>(g * R + i, h) * 2);
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
                                   s + j * F32 + seq_off_h<TH>(g * R + r, h) * 4, 64);
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

// The same in two steps for N tiles with one barrier (I2M_SEQ_ONE; scratch slot n G + g for tile n's group g):
// seq_pre issues the committed rows' reads straight from the state tensor tiles (NOC addresses s0..s2 of tile n),
// beside the caller's own reads; after the caller's barrier, seq_fin stages the inputs and rewrites the N tiles' states.
template <uint32_t N>
constexpr bool seq_fits() {
    return 8 * (N * SEQ_G - 1) + SEQ_R + 1 < 30;  // (rows 30 / 31: the accepted rows)
}

inline void seq_pre(uint64_t s0, uint64_t s1, uint64_t s2, uint32_t n, uint32_t scratch,
                    volatile tt_l1_ptr uint32_t* acc) {
    constexpr uint32_t R = SEQ_R, G = SEQ_G;
    for (uint32_t g = 0; g < G; ++g) {
        const uint32_t ra = g * R + acc[g], z = (n * G + g) * 8;
        for (uint32_t h = 0; h < 2; ++h) {
            noc_async_read(s0 + seq_off(ra, h) * 4, scratch + seq_off(z, h) * 4, 64);
            noc_async_read(s1 + seq_off(ra, h) * 4, scratch + seq_off(z + 1, h) * 4, 64);
            noc_async_read(s2 + seq_off(ra, h) * 4, scratch + seq_off(z + 2, h) * 4, 64);
        }
    }
}

// s / x: tile 0's 3 state tiles / input tile, tile n's at + n s_step / x_step
template <uint32_t XE, uint32_t TH, uint32_t N>
inline void seq_fin(uint32_t s, uint32_t s_step, uint32_t x, uint32_t x_step, uint32_t scratch) {
    constexpr uint32_t R = SEQ_R, G = SEQ_G, F32 = TH * 128;
    static_assert(R + 2 <= 8 && G <= 4 && R * G <= TH && seq_fits<N>(), "verify rows");
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t g = 0; g < G; ++g) {
            const uint32_t z = (n * G + g) * 8;
            for (uint32_t i = 0; i + 1 < R; ++i) {
                for (uint32_t h = 0; h < 2; ++h) {
                    auto dst = reinterpret_cast<tt_l1_ptr uint32_t*>(scratch + seq_off(z + 3 + i, h) * 4);
                    if constexpr (XE == 4) {
                        auto src = reinterpret_cast<tt_l1_ptr uint32_t*>(x + n * x_step + seq_off_h<TH>(g * R + i, h) * 4);
                        for (uint32_t e = 0; e < 16; ++e) {
                            dst[e] = src[e];
                        }
                    } else {  // (two bf16 per load: element 2 e in the low half)
                        auto src = reinterpret_cast<tt_l1_ptr uint32_t*>(x + n * x_step + seq_off_h<TH>(g * R + i, h) * 2);
                        for (uint32_t e = 0; e < 8; ++e) {
                            const uint32_t v = src[e];
                            dst[2 * e] = v << 16;
                            dst[2 * e + 1] = v & 0xFFFF0000u;
                        }
                    }
                }
            }
        }
    }
    // state j rows g R .. g R + R - 1 = z rows j .. j + R - 1: one read per half (the rows of a face are contiguous; a
    // slot's 8 z rows sit in one face)
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t g = 0; g < G; ++g) {
            const uint32_t z = (n * G + g) * 8;
            const bool one = TH < 32 || (g * R) >> 4 == (g * R + R - 1) >> 4;
            for (uint32_t j = 0; j < 3; ++j) {
                for (uint32_t h = 0; h < 2; ++h) {
                    const uint32_t d = s + n * s_step + j * F32;
                    if (one) {
                        noc_async_read(get_noc_addr(scratch + seq_off(z + j, h) * 4), d + seq_off_h<TH>(g * R, h) * 4,
                                       R * 64);
                    } else {
                        for (uint32_t r = 0; r < R; ++r) {
                            noc_async_read(get_noc_addr(scratch + seq_off(z + r + j, h) * 4),
                                           d + seq_off_h<TH>(g * R + r, h) * 4, 64);
                        }
                    }
                }
            }
        }
    }
    noc_async_read_barrier();
}
