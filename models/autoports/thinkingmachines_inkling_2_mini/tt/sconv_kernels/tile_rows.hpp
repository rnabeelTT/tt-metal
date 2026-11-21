// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Row access in a bf16 32x32 tile (4 faces of 16x16: f0 rows 0-15 cols 0-15, f1 rows 0-15 cols 16-31, f2 rows 16-31
// cols 0-15, f3 rows 16-31 cols 16-31; each face row 16 contiguous bf16), and fp32 <-> bf16 rows (the short conv's
// values are bf16-exact: the fp32 state rows hold bf16 inputs, so the conversions are exact).
#pragma once
#include <stdint.h>

namespace sconv {

// word (2 x bf16) offset of row r's first 8 words in face (r / 16) * 2, the other 8 in the next face (+128 words)
inline uint32_t row_word(uint32_t r) { return (r >> 4) * 256 + (r & 15) * 8; }

inline void copy_row(volatile uint32_t* dst, uint32_t rd, volatile uint32_t* src, uint32_t rs) {
    const uint32_t d = row_word(rd), s = row_word(rs);
    for (uint32_t i = 0; i < 8; ++i) {
        dst[d + i] = src[s + i];
        dst[d + 128 + i] = src[s + 128 + i];
    }
}

inline void zero_row(volatile uint32_t* dst, uint32_t rd) {
    const uint32_t d = row_word(rd);
    for (uint32_t i = 0; i < 8; ++i) {
        dst[d + i] = 0;
        dst[d + 128 + i] = 0;
    }
}

// 32 fp32 values (one state row's slice) -> tile row rd as bf16 (the top 16 bits: exact for bf16 values)
inline void f32_to_row(volatile uint32_t* dst, uint32_t rd, volatile uint32_t* f32) {
    const uint32_t d = row_word(rd);
    for (uint32_t i = 0; i < 8; ++i) {
        dst[d + i] = (f32[2 * i] >> 16) | (f32[2 * i + 1] & 0xffff0000u);
        dst[d + 128 + i] = (f32[16 + 2 * i] >> 16) | (f32[16 + 2 * i + 1] & 0xffff0000u);
    }
}

// tile row rs (bf16) -> 32 fp32 values
inline void row_to_f32(volatile uint32_t* f32, volatile uint32_t* src, uint32_t rs) {
    const uint32_t s = row_word(rs);
    for (uint32_t i = 0; i < 8; ++i) {
        const uint32_t a = src[s + i], b = src[s + 128 + i];
        f32[2 * i] = a << 16;
        f32[2 * i + 1] = a & 0xffff0000u;
        f32[16 + 2 * i] = b << 16;
        f32[16 + 2 * i + 1] = b & 0xffff0000u;
    }
}

}  // namespace sconv
