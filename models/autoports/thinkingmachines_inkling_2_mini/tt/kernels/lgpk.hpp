// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Packed mix logits (tt/lgx.py FOLD, LG_PK = 1): logit e is column e % 32 of tile e / 32 (not a whole replicated tile
// e). A reader reads that tile's rows where the replicated tile would go, then lg_bcast copies column e % 32 over the
// 32 columns of rows 0 .. T-1: the values tt/kernels/lgx.cpp writes, without its program.

#pragma once

#include <stdint.h>

#ifndef LG_PK
#define LG_PK 0
#endif

inline uint32_t lg_tile(uint32_t e) { return LG_PK ? e / 32 : e; }

// dst: a TH x 32 fp32 CB tile (faces min(TH, 16) x 16; faces 2 / 3 after faces 0 / 1 when TH = 32)
template <uint32_t T, uint32_t TH>
inline void lg_bcast(uint32_t dst, uint32_t e) {
    if constexpr (LG_PK) {
        constexpr uint32_t F1 = (TH < 16 ? TH : 16) * 16;  // (words)
        const uint32_t j = e % 32, c0 = (j < 16 ? 0 : F1) + j % 16;
        for (uint32_t r = 0; r < T; ++r) {
            volatile tt_l1_ptr uint32_t* w =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst) + (r < 16 ? 0 : 2 * F1) + (r % 16) * 16;
            const uint32_t v = w[c0];
            for (uint32_t c = 0; c < 16; ++c) {
                w[c] = v;
                w[F1 + c] = v;
            }
        }
    }
}
