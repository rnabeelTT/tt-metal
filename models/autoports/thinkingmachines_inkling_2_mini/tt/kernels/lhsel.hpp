// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// One row's top 32 by (bf16(x - m) descending, column ascending) from the LM head cores' fp32 lists (tt/sampling.py
// HeadTopK): the key is monotone in x, so the lists merge in fp32 order; RowsTopK's order and values.

#pragma once

#include <stdint.h>

#include "lhkey.hpp"
#include "lhlist.hpp"

constexpr uint32_t LH_K = 32;

// a bf16 value's order (RowsTopK's: -0 just below +0)
static inline uint32_t lh_ord16(uint32_t k) { return (k & 0x8000) ? (~k & 0xFFFF) : (k | 0x8000); }

struct LhBest {  // the best LH_K (key order descending, then the lower column), one entry per column
    uint32_t key[LH_K], col[LH_K];
    uint32_t n;
    uint32_t last() const { return n < LH_K ? 0 : key[LH_K - 1]; }  // (keys below it cannot enter)
    void add(uint32_t k, uint32_t c) {
        if (n == LH_K && (k < key[LH_K - 1] || (k == key[LH_K - 1] && c > col[LH_K - 1]))) {
            return;
        }
        for (uint32_t j = 0; j < n; ++j) {
            if (col[j] == c) {
                return;
            }
        }
        uint32_t i = n < LH_K ? n++ : LH_K - 1;
        while (i > 0 && (key[i - 1] < k || (key[i - 1] == k && col[i - 1] > c))) {
            key[i] = key[i - 1];
            col[i] = col[i - 1];
            --i;
        }
        key[i] = k;
        col[i] = c;
    }
};

// pages: NC list pages (lhlist.hpp: sorted values, columns, count, thr, dropped); mu: fp32 bits of the shift. The lists
// are merged in fp32 order until the keys fall below the 32nd; a core that dropped values whose key may reach it
// rescans: rescan(c, thr, best) offers its dropped values (<= thr) to best. ws: NC words. -> best; returns the rescans.
template <uint32_t NC, typename Rescan>
static inline uint32_t lh_select(const uint32_t* pages, uint32_t mu, LhBest& best, uint32_t* ptr, Rescan rescan) {
    constexpr uint32_t PW = LH_PAGE / 4;
    auto S = [&](uint32_t c, uint32_t i) { return static_cast<int32_t>(pages[c * PW + i]); };
    auto C = [&](uint32_t c, uint32_t i) { return pages[c * PW + LH_KL + i]; };
    auto cnt = [&](uint32_t c) { return pages[c * PW + 2 * LH_KL]; };
    for (uint32_t c = 0; c < NC; ++c) {
        ptr[c] = 0;
    }
    best.n = 0;
    while (true) {
        int32_t b = -1;  // the list holding the next value (fp32 order, then the lower column)
        for (uint32_t c = 0; c < NC; ++c) {
            if (ptr[c] < cnt(c) &&
                (b < 0 || S(c, ptr[c]) > S(b, ptr[b]) || (S(c, ptr[c]) == S(b, ptr[b]) && C(c, ptr[c]) < C(b, ptr[b])))) {
                b = c;
            }
        }
        if (b < 0) {
            break;
        }
        const uint32_t k = lh_ord16(lh_key(lh_unsord(S(b, ptr[b])), mu));
        if (k < best.last()) {
            break;
        }
        best.add(k, C(b, ptr[b]));
        ++ptr[b];
    }
    uint32_t nres = 0;
    for (uint32_t c = 0; c < NC; ++c) {
        const int32_t thr = static_cast<int32_t>(pages[c * PW + 2 * LH_KL + 1]);
        if (pages[c * PW + 2 * LH_KL + 2] && lh_ord16(lh_key(lh_unsord(thr), mu)) >= best.last()) {
            rescan(c, thr, best);
            ++nres;
        }
    }
    return nres;
}

static inline uint16_t lh_unord16(uint32_t o) { return (o & 0x8000) ? (o & 0x7FFF) : (~o & 0xFFFF); }

// the smallest signed-order x whose key order is >= v32 (keys are monotone in x), searched in [lo, hi]
static inline int32_t lh_xlo(int32_t lo, int32_t hi, uint32_t mu, uint32_t v32) {
    while (lo < hi) {
        const int32_t mid = lo + static_cast<int32_t>((static_cast<uint32_t>(hi) - static_cast<uint32_t>(lo)) / 2);
        if (lh_ord16(lh_key(lh_unsord(mid), mu)) >= v32) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return lo;
}
