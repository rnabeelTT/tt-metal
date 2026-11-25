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

// GMM_GS: a group slice takes the real groups (ids < E, sorted first) [gr q0 / Q, gr q1 / Q): its share [q0, q1) of Q
// (tt/gmm.py: q0, Q << 16 | q1; equal shares q = s, s + 1 of S) whatever the experts' loads; an expert's groups may
// span slices (each loads its weight block).
template <typename F>
inline void gmm_gs(F id, uint32_t G, uint32_t E, uint32_t q0, uint32_t qq, uint32_t& g0, uint32_t& g1) {
    uint32_t lo = 0, hi = G;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) / 2;
        if (id(mid) < E) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    const uint32_t Q = qq >> 16, q1 = qq & 0xFFFF;
    g0 = lo * q0 / Q;
    g1 = lo * q1 / Q;
}

// GMM_W2 = W2 (act): the gate|up pairs continue in a second weight tensor ([gate | up], W2 column tiles each) after
// the first's Wt pairs: block column j (j < h: gate, else up) of pair c0 + j % h -> its tensor and column tile
// (row stride: 2 Wt or 2 W2 tiles).
template <uint32_t Wt, uint32_t W2, uint32_t h>
inline uint32_t gmm_w2_col(uint32_t c0, uint32_t j, bool& second) {
    const uint32_t p = c0 + (j < h ? j : j - h), up = j < h ? 0 : 1;
    second = W2 > 0 && p >= Wt;
    return second ? p - Wt + up * W2 : p + up * Wt;
}
