// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Lane-aware score schedule (see ../__init__.py LaneSelOp.score), shared by the reader and the writer: both parse the
// same lmeta rows (L1, [32][16] uint32: lane group base lo, visible groups vis, cache base, selects) into the same
// units. A sub-run = up to 2 consecutive rows with the same lo (one packed q tile: 16 heads x 2 rows). Compute units:
// (sub-run, key tile kt < vt) with vt = ceil(max selecting vis / 32); fill units (-inf only): kt in [vt, VT), VT the
// largest vt. Core c of NC takes the contiguous ranges [c C / NC, (c + 1) C / NC) of each list.
#pragma once

#include <stdint.h>

// this core's index in LaneSelOp.score's row-major grid (LS_GX = grid width)
inline uint32_t ls_core() { return static_cast<uint32_t>(get_absolute_logical_y()) * LS_GX + get_absolute_logical_x(); }

struct LsSched {
    uint32_t nrun = 0, VT = 0, C = 0, F = 0;
    uint32_t r0[32], n[32], lo[32], vt[32];
};

inline void ls_build(volatile tt_l1_ptr uint32_t* m, LsSched& s) {
    uint32_t r = 0;
    while (r < 32) {
        const uint32_t lo = m[r * 16];
        uint32_t n = 1;
        while (r + n < 32 && n < 2 && m[(r + n) * 16] == lo) {
            ++n;
        }
        uint32_t vt = 0;
        for (uint32_t j = 0; j < n; ++j) {
            if (m[(r + j) * 16 + 3]) {
                const uint32_t t = (m[(r + j) * 16 + 1] + 31) / 32;
                vt = t > vt ? t : vt;
            }
        }
        const uint32_t i = s.nrun++;
        s.r0[i] = r;
        s.n[i] = n;
        s.lo[i] = lo;
        s.vt[i] = vt;
        s.VT = vt > s.VT ? vt : s.VT;
        s.C += vt;
        r += n;
    }
    for (uint32_t i = 0; i < s.nrun; ++i) {
        s.F += s.VT - s.vt[i];
    }
}

// Unit cursor over one list (fill = false: [0, vt) of each sub-run; true: [vt, VT)).
struct LsCursor {
    uint32_t run = 0, kt = 0;
};

inline uint32_t ls_len(const LsSched& s, uint32_t i, bool fill) { return fill ? s.VT - s.vt[i] : s.vt[i]; }

inline LsCursor ls_seek(const LsSched& s, uint32_t u, bool fill) {
    LsCursor c;
    while (c.run < s.nrun && u >= ls_len(s, c.run, fill)) {
        u -= ls_len(s, c.run, fill);
        ++c.run;
    }
    c.kt = u + (fill ? s.vt[c.run < s.nrun ? c.run : 0] : 0);
    return c;
}

inline void ls_next(const LsSched& s, LsCursor& c, bool fill) {
    ++c.kt;
    const uint32_t end = fill ? s.VT : s.vt[c.run];
    if (c.kt >= end) {
        ++c.run;
        while (c.run < s.nrun && ls_len(s, c.run, fill) == 0) {
            ++c.run;
        }
        c.kt = (c.run < s.nrun && fill) ? s.vt[c.run] : 0;
    }
}

inline void ls_range(uint32_t total, uint32_t core, uint32_t ncores, uint32_t& b, uint32_t& e) {
    b = total * core / ncores;
    e = total * (core + 1) / ncores;
}
