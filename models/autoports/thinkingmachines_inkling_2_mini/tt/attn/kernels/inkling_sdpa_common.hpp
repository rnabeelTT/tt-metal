// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Inkling SDPA decode (tt/attn/inkling_sdpa.py), derived from the shared-KV rows SDPA decode (qwen36
// attn_decode/rows_sdpa.py): the per-row causal limits every kernel derives the same way from the cur_pos stick, so
// the reader, writer and compute agree on the chunk split, on which chunks carry a mask (causal edge, sliding-window
// edge) and on which carry a relative-position bias (the band of distances < E).
//
// Q tile row rho (rho < R * G) is query head (rho % G) of kv head g of verify row rho / G; it attends positions
// [0, lim[rho / G]]. Rows past R * G are padding (zero Q): they take max_pos. A skipped row (cur_pos < 0, the stock
// op's -1) also takes max_pos (finite, never written out).

#pragma once

#include <stdint.h>

#include <tuple>

constexpr uint32_t ROWS_SDPA_MAX_ROWS = 32;

// lim[0..R) = each row's last attended position; returns false when every row is skipped. max_pos = the newest
// position any row attends (it sizes the chunk split), min_lim = the oldest last position (a chunk whose last position
// is past it needs a mask).
template <uint32_t R, typename GetPos>
inline bool rows_sdpa_limits(GetPos get_pos, uint32_t* lim, uint32_t& max_pos, uint32_t& min_lim) {
    bool any = false;
    max_pos = 0;
    for (uint32_t j = 0; j < R; ++j) {
        const uint32_t p = get_pos(j);
        lim[j] = p;
        if (static_cast<int32_t>(p) >= 0) {
            any = true;
            max_pos = p > max_pos ? p : max_pos;
        }
    }
    if (!any) {
        return false;
    }
    min_lim = max_pos;
    for (uint32_t j = 0; j < R; ++j) {
        if (static_cast<int32_t>(lim[j]) < 0) {
            lim[j] = max_pos;
        }
        min_lim = lim[j] < min_lim ? lim[j] : min_lim;
    }
    return true;
}

// chunk k_chunk covers positions [k_chunk * chunk_positions, (k_chunk + 1) * chunk_positions)
inline bool rows_sdpa_chunk_masked(uint32_t k_chunk, uint32_t chunk_positions, uint32_t min_lim) {
    return (k_chunk + 1) * chunk_positions - 1 > min_lim;
}

// Sliding window W (0 = none): row j attends [lo_j, lim_j] with lo_j = lim_j + 1 - W (clamped at 0). max_lo = the
// largest lower bound (a chunk starting before it has positions some row excludes), min_lo = the smallest (the first
// position any row attends).
inline uint32_t inkling_lo(uint32_t lim, uint32_t W) { return (W == 0 || lim + 1 < W) ? 0 : lim + 1 - W; }

// mask block needed: the causal edge (a position past the oldest row limit) or the window edge
inline bool inkling_chunk_masked(
    uint32_t k_chunk, uint32_t chunk_positions, uint32_t min_lim, uint32_t W, uint32_t max_lo) {
    return rows_sdpa_chunk_masked(k_chunk, chunk_positions, min_lim) || (W != 0 && k_chunk * chunk_positions < max_lo);
}

// relative-bias block needed: some position of the chunk within E of some row (E = rel_extent; the bias is zero at
// distances >= E, so far chunks of a global layer run the stock path). A masked chunk is always a band chunk
// (causal edge: distance 0; window edge: E == W on sliding layers), so the mask always rides the bias matmul.
inline bool inkling_chunk_band(uint32_t k_chunk, uint32_t chunk_positions, uint32_t min_lim, uint32_t E) {
    return (k_chunk + 1) * chunk_positions - 1 + E > min_lim;
}

// bf16 32x32 tile: byte offset of tile row `row`'s left face row (its right half is 512 bytes further)
inline uint32_t rows_sdpa_row_off(uint32_t row) { return (row >> 4) * 1024 + (row & 15) * 32; }

// tile rows [src, src + n) -> [dst, dst + n) as runs that stay inside one face pair on both sides (a run of k rows is
// contiguous: k * 32 bytes per face); fn(src_row, dst_row, k)
template <typename Fn>
inline void rows_sdpa_row_runs(uint32_t src, uint32_t dst, uint32_t n, Fn fn) {
    while (n) {
        const uint32_t s_left = 16 - (src & 15), d_left = 16 - (dst & 15);
        uint32_t k = n < s_left ? n : s_left;
        k = k < d_left ? k : d_left;
        fn(src, dst, k);
        src += k;
        dst += k;
        n -= k;
    }
}

// SDPA_TRIM (decode: one unit per core): a core's LAST chunk covers only its key tiles some row attends, [tb, tb + n):
// the newest chunk's tiles past max_pos and the window's first chunk's tiles before min_lo are fully masked (they add
// exact zeros), so they are neither read nor computed. Only the last chunk of a core is trimmed: every CB block before
// it keeps the full size (a smaller block last never wraps a CB).
#ifndef SDPA_TRIM
#define SDPA_TRIM 0
#endif
#ifndef SDPA_BALANCE
#define SDPA_BALANCE 0
#endif
// SDPA_BALANCE (with SDPA_TRIM): when every core holds one chunk (k_num_chunks <= cores per unit), the live tiles
// [min_lo / 32, max_pos / 32] are split evenly over the same number of chunks (k_rel = k_chunk - the window's first
// chunk): the newest chunk otherwise holds a few live tiles and the others a full chunk each
inline void inkling_chunk_tiles(
    uint32_t k_chunk, uint32_t Sk, uint32_t max_pos, uint32_t min_lo, bool last, uint32_t& tb, uint32_t& n,
    uint32_t k_num_chunks = 0, uint32_t window_start_chunk = 0, uint32_t cores = 0) {
    if (SDPA_TRIM && SDPA_BALANCE && k_num_chunks <= cores) {
        const uint32_t first = min_lo >> 5, T = (max_pos >> 5) + 1 - first, k = k_chunk - window_start_chunk;
        tb = first + k * T / k_num_chunks;
        n = first + (k + 1) * T / k_num_chunks - tb;
        return;
    }
    tb = k_chunk * Sk;
    uint32_t te = tb + Sk;
    if (SDPA_TRIM && last) {
        const uint32_t a = min_lo >> 5, b = (max_pos >> 5) + 1;
        tb = a > tb ? a : tb;
        te = b < te ? b : te;
    }
    n = te - tb;
}

// the mask / band tests on a chunk's tiles [tb, tb + n) (inkling_chunk_masked / inkling_chunk_band on whole chunks)
inline bool inkling_tiles_masked(uint32_t tb, uint32_t n, uint32_t min_lim, uint32_t W, uint32_t max_lo) {
    return (tb + n) * 32 - 1 > min_lim || (W != 0 && tb * 32 < max_lo);
}
inline bool inkling_tiles_band(uint32_t tb, uint32_t n, uint32_t min_lim, uint32_t E) {
    return (tb + n) * 32 - 1 + E > min_lim;
}

// SDPA_MU_MIN (multi-user decode split over cores, MU_SPLIT): a unit with at most SDPA_MU_MIN chunks runs on its
// root core alone, every chunk in order (the unsplit program's arithmetic: short contexts stay bitwise); longer ones
// split over the unit's cores (the stock workload). cores_eff = the cores of the unit with work allowed.
#ifndef SDPA_MU_MIN
#define SDPA_MU_MIN 0
#endif
inline std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> inkling_workload(
    uint32_t max_pos, uint32_t core, uint32_t cores, uint32_t chunk_size, uint32_t W_eff, uint32_t& cores_eff) {
    auto w = get_workload_for_core(max_pos, 0, core, cores, chunk_size, W_eff);
    cores_eff = cores;
    if (SDPA_MU_MIN && std::get<1>(w) <= SDPA_MU_MIN) {
        cores_eff = 1;
        if (core == 0) {
            w = get_workload_for_core(max_pos, 0, 0, 1, chunk_size, W_eff);
        } else {
            std::get<3>(w) = std::get<2>(w);
        }
    }
    return w;
}
