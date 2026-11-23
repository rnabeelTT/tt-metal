// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The cross-core reductions' gather layout (tt/post.py red_sel): core c's partial row -> tile c / (4 K), row
// 8 ((c % 4 K) / K) + c % K. The FPU sums a matmul's inner dim in 8-row groups at reduced precision (the group sums
// then accumulate in fp32), so at most K partials share a group; the unused rows are zeroed (0 x NaN is NaN).

#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

inline uint32_t red_row_off(uint32_t i) { return (i < 16 ? 0u : 2048u) + (i % 16) * 64; }

// byte offset of core c's row in a part's gather tiles
template <uint32_t K>
inline uint32_t red_off(uint32_t c) {
    return (c / (4 * K)) * 4096 + red_row_off(8 * ((c % (4 * K)) / K) + c % K);
}

// zero the unused rows of NQ gather tiles at base (local L1) from the zero block z (>= 512 B)
template <uint32_t K, uint32_t C, uint32_t NQ>
inline void red_zero(uint32_t z, uint32_t base) {
    for (uint32_t k = 0; k < NQ; ++k) {
        for (uint32_t grp = 0; grp < 4; ++grp) {
            const uint32_t first = k * 4 * K + grp * K;
            const uint32_t used = first >= C ? 0 : (C - first < K ? C - first : K);
            if (used < 8) {
                const uint32_t o = base + k * 4096 + red_row_off(8 * grp + used), n = (8 - used) * 64;
                noc_async_write(z, get_noc_addr(o), n);
                noc_async_write(z, get_noc_addr(o + 1024), n);
            }
        }
    }
}
