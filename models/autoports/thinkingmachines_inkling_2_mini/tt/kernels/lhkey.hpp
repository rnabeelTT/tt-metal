// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The sampling tail's candidate value in integer code: bf16(fp32(x - m)), both roundings to nearest even (as the
// device's fp32 subtract + bf16 typecast). x, m: finite fp32 bits.

#pragma once

#include <stdint.h>

// signed order of fp32 bits: s(a) < s(b) iff a < b (-0 just below +0)
static inline int32_t lh_sord(uint32_t u) { return static_cast<int32_t>(u ^ ((static_cast<int32_t>(u) >> 31) & 0x7FFFFFFF)); }
static inline uint32_t lh_unsord(int32_t s) { return static_cast<uint32_t>(s ^ ((s >> 31) & 0x7FFFFFFF)); }

static inline int32_t lh_msb(uint32_t v) {  // -1 for 0
    if (v == 0) {
        return -1;
    }
    int32_t p = 0;
    for (uint32_t k = 16; k; k >>= 1) {
        if (v >> k) {
            v >>= k;
            p += k;
        }
    }
    return p;
}

static inline uint32_t lh_f32_sub(uint32_t a, uint32_t b) {
    b ^= 0x80000000u;
    uint32_t ea = (a >> 23) & 0xFF, eb = (b >> 23) & 0xFF;
    uint32_t ma = a & 0x7FFFFF, mb = b & 0x7FFFFF;
    if (ea) {
        ma |= 0x800000;
    } else {
        ea = 1;
    }
    if (eb) {
        mb |= 0x800000;
    } else {
        eb = 1;
    }
    if (ma == 0) {
        return mb ? b : (a & b & 0x80000000u);  // (0 - 0: +0 unless both -0)
    }
    if (mb == 0) {
        return a;
    }
    // exact sum on a common exponent (an operand more than 38 binades below the other cannot move the rounding)
    if (ea > eb + 38) {
        return a;
    }
    if (eb > ea + 38) {
        return b;
    }
    const uint32_t e0 = ea < eb ? ea : eb;
    int64_t sa = static_cast<int64_t>(ma) << (ea - e0), sb = static_cast<int64_t>(mb) << (eb - e0);
    if (a & 0x80000000u) {
        sa = -sa;
    }
    if (b & 0x80000000u) {
        sb = -sb;
    }
    int64_t s = sa + sb;
    if (s == 0) {
        return 0;
    }
    uint32_t sign = 0;
    if (s < 0) {
        sign = 0x80000000u;
        s = -s;
    }
    uint64_t v = static_cast<uint64_t>(s);
    int32_t p = lh_msb(static_cast<uint32_t>(v >> 32));  // msb position
    p = p >= 0 ? p + 32 : lh_msb(static_cast<uint32_t>(v));
    int32_t e = static_cast<int32_t>(e0) + p - 23;  // biased exponent of the normalized result
    uint64_t r;
    if (p > 23) {
        const uint32_t sh = p - 23;
        const uint64_t half = 1ull << (sh - 1), rem = v & ((1ull << sh) - 1);
        r = v >> sh;
        if (rem > half || (rem == half && (r & 1))) {
            ++r;
        }
        if (r >> 24) {
            r >>= 1;
            ++e;
        }
    } else {
        r = v << (23 - p);
    }
    if (e <= 0 || e >= 255) {
        return sign;  // (out of the logits' range: never reached)
    }
    return sign | (static_cast<uint32_t>(e) << 23) | (static_cast<uint32_t>(r) & 0x7FFFFF);
}

static inline uint32_t lh_bf16(uint32_t u) { return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16; }

static inline uint32_t lh_key(uint32_t x, uint32_t m) { return lh_bf16(lh_f32_sub(x, m)); }
