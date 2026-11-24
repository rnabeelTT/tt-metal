// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Grouped expert matmul (tt/gmm.py), GMM_BAL = R > 0: the expert range [e0, e1) of group slice s of S from the sorted
// ids, split by cost (R per expert with groups + 1 per group: weight blocks are loaded only for experts with groups);
// every kernel of every core computes the same ranges (contiguous, in expert order).

#pragma once

#include <stdint.h>

template <typename F>
inline void gmm_bal(F id, uint32_t G, uint32_t E, uint32_t s, uint32_t S, uint32_t R, uint32_t& e0, uint32_t& e1) {
    uint32_t total = 0, prev = 0xFFFFFFFF;
    for (uint32_t g = 0; g < G; ++g) {
        const uint32_t e = id(g);
        if (e >= E) {
            break;
        }
        total += (e != prev ? R : 0) + 1;
        prev = e;
    }
    e0 = e1 = E;
    uint32_t cum = 0;
    prev = 0xFFFFFFFF;
    for (uint32_t g = 0; g < G; ++g) {
        const uint32_t e = id(g);
        if (e >= E) {
            break;
        }
        if (e != prev) {
            const uint32_t sl = cum * S / total;  // (the slice of the expert's first group)
            if (sl > s) {
                e1 = e;
                break;
            }
            if (sl == s && e0 == E) {
                e0 = e;
            }
            cum += R;
            prev = e;
        }
        cum += 1;
    }
    e0 = e0 < e1 ? e0 : e1;
}
