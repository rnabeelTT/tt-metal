// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Shared-KV rows SDPA decode (attn_decode/rows_sdpa.py): the per-row causal limits every kernel derives the same way
// from the cur_pos stick, so the reader, writer and compute agree on the chunk split and on which chunks carry a mask.
//
// Q tile row rho (rho < R * G) is query head (rho % G) of kv head g of verify row rho / G; it attends positions
// [0, lim[rho / G]]. Rows past R * G are padding (zero Q): they take max_pos. A skipped row (cur_pos < 0, the stock
// op's -1) also takes max_pos (finite, never written out).

#pragma once

#include <stdint.h>

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
