// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The LM head writer's per-row candidate list (head_writer.cpp HEAD_TOPK): the values above a threshold are appended
// unsorted; a full buffer keeps its LH_KL best (the threshold rises to the LH_KL-th value: the values above it, then
// ties at it while they fit). Invariant: every value above `thr` is kept, every dropped value is <= thr. At the end
// the list (<= LH_KL) is sorted best first (value, then the lower column) and the page holds
// [LH_KL values][LH_KL columns][count, thr, has-dropped, pad] (signed-order values: lhkey.hpp lh_sord).

#pragma once

#include <stdint.h>

#ifndef LH_HEAP
#define LH_HEAP 0
#endif

constexpr uint32_t LH_KL = 64;
constexpr uint32_t LH_CAP = 192;            // buffer entries per row
constexpr uint32_t LH_PAGE = LH_KL * 8 + 64;  // list page bytes
constexpr uint32_t LH_ROW_WORDS = 2 * LH_CAP + 16;  // (64 B multiple: the pages after the rows stay aligned)

struct LhRow {  // in L1: S[LH_CAP], C[LH_CAP], n, thr, dropped, pad
    int32_t* S() { return reinterpret_cast<int32_t*>(this); }
    uint32_t* C() { return reinterpret_cast<uint32_t*>(this) + LH_CAP; }
    uint32_t& n() { return reinterpret_cast<uint32_t*>(this)[2 * LH_CAP]; }
    int32_t& thr() { return reinterpret_cast<int32_t*>(this)[2 * LH_CAP + 1]; }
    uint32_t& dropped() { return reinterpret_cast<uint32_t*>(this)[2 * LH_CAP + 2]; }
};

static inline void lh_row_init(LhRow* r) {
    r->n() = 0;
    r->thr() = INT32_MIN;
    r->dropped() = 0;
}

// the k-th largest (0-based) of a[0 .. n) (a is reordered)
static inline int32_t lh_kth(int32_t* a, uint32_t n, uint32_t k) {
    uint32_t lo = 0, hi = n - 1;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        int32_t x = a[lo], y = a[mid], z = a[hi];  // median of three as the pivot
        const int32_t p = x > y ? (y > z ? y : (x > z ? z : x)) : (x > z ? x : (y > z ? z : y));
        uint32_t i = lo, j = hi;
        while (i <= j) {  // descending: [lo, j] >= p, [i, hi] <= p
            while (a[i] > p) {
                ++i;
            }
            while (a[j] < p) {
                --j;
            }
            if (i <= j) {
                const int32_t t = a[i];
                a[i] = a[j];
                a[j] = t;
                ++i;
                if (j == 0) {
                    break;
                }
                --j;
            }
        }
        if (k <= j && j != static_cast<uint32_t>(-1)) {
            hi = j;
        } else if (k >= i) {
            lo = i;
        } else {
            return a[k];
        }
    }
    return a[lo];
}

// keep the LH_KL best (scratch: LH_CAP words)
static inline void lh_row_compact(LhRow* r, int32_t* scratch) {
    const uint32_t n = r->n();
    if (n <= LH_KL) {
        return;
    }
    int32_t* S = r->S();
    uint32_t* C = r->C();
    for (uint32_t i = 0; i < n; ++i) {
        scratch[i] = S[i];
    }
    const int32_t t = lh_kth(scratch, n, LH_KL - 1);
    uint32_t m = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (S[i] > t) {
            S[m] = S[i];
            C[m] = C[i];
            ++m;
        }
    }
    for (uint32_t i = 0; i < n && m < LH_KL; ++i) {  // (ties at t: the lower columns first, as they arrived)
        if (S[i] == t) {
            S[m] = S[i];
            C[m] = C[i];
            ++m;
        }
    }
    r->n() = m;
    r->thr() = t;
    r->dropped() = 1;
}

#if LH_HEAP
// LH_HEAP: a min-heap of the LH_KL best instead (root: the worst, by value then the higher column; thr: its value once
// full)
static inline bool lh_worse(int32_t sa, uint32_t ca, int32_t sb, uint32_t cb) { return sa < sb || (sa == sb && ca > cb); }

static inline void lh_row_add(LhRow* r, int32_t s, uint32_t col, int32_t*) {
    int32_t* S = r->S();
    uint32_t* C = r->C();
    uint32_t n = r->n();
    uint32_t i;
    if (n < LH_KL) {
        i = n;
        while (i > 0 && lh_worse(s, col, S[(i - 1) / 2], C[(i - 1) / 2])) {
            S[i] = S[(i - 1) / 2];
            C[i] = C[(i - 1) / 2];
            i = (i - 1) / 2;
        }
        r->n() = ++n;
    } else {
        i = 0;
        while (true) {
            uint32_t l = 2 * i + 1;
            if (l >= LH_KL) {
                break;
            }
            if (l + 1 < LH_KL && lh_worse(S[l + 1], C[l + 1], S[l], C[l])) {
                ++l;
            }
            if (!lh_worse(S[l], C[l], s, col)) {
                break;
            }
            S[i] = S[l];
            C[i] = C[l];
            i = l;
        }
    }
    S[i] = s;
    C[i] = col;
    if (n == LH_KL) {
        r->thr() = S[0];
        r->dropped() = 1;  // (from here on a value can be dropped)
    }
}
#else
static inline void lh_row_add(LhRow* r, int32_t s, uint32_t col, int32_t* scratch) {
    uint32_t n = r->n();
    r->S()[n] = s;
    r->C()[n] = col;
    r->n() = ++n;
    if (n == LH_CAP) {
        lh_row_compact(r, scratch);
    }
}
#endif

// compact to <= LH_KL and sort best first; -> the page (LH_PAGE bytes) at out
static inline void lh_row_finish(LhRow* r, int32_t* scratch, uint32_t* out) {
    lh_row_compact(r, scratch);
    const uint32_t n = r->n();
    int32_t* S = r->S();
    uint32_t* C = r->C();
    int32_t* oS = reinterpret_cast<int32_t*>(out);
    uint32_t* oC = out + LH_KL;
    for (uint32_t j = 0; j < n; ++j) {  // insertion sort into the page
        const int32_t s = S[j];
        const uint32_t c = C[j];
        uint32_t i = j;
        while (i > 0 && (oS[i - 1] < s || (oS[i - 1] == s && oC[i - 1] > c))) {
            oS[i] = oS[i - 1];
            oC[i] = oC[i - 1];
            --i;
        }
        oS[i] = s;
        oC[i] = c;
    }
    out[2 * LH_KL] = n;
    out[2 * LH_KL + 1] = static_cast<uint32_t>(r->thr());
    out[2 * LH_KL + 2] = r->dropped();
}
