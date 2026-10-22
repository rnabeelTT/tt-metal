// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Router top-k + softmax on a data-movement core (no FPU), for the Qwen3.5-MoE decode router
// (moe_decode/topk.py). Top-k by order-preserving integer keys of the BF16 logits (ties to the lower id, one pass
// of insertion into a sorted k-list), softmax over the selected logits in 64-bit fixed point (exp2 = integer shift x
// degree-5 polynomial, ~1e-5 relative), rounded to BF16. softmax over all experts -> top-k -> renormalize equals
// top-k of the logits -> softmax over the k selected (the global partition function cancels).
// Fixed-point helpers from the gpt-oss fused decode router (models/demos/gpt_oss/tt/fused_decode/kernels).

#pragma once

#include <stdint.h>

namespace moe_topk {
// BF16 bits -> Q16.16 fixed point (|x| < 2^14).
inline int32_t bf16_to_q16(uint16_t b) {
    const int32_t e = (b >> 7) & 0xFF;
    if (e == 0) {
        return 0;
    }
    const int32_t m = (b & 0x7F) | 0x80;
    const int32_t shift = e - 118;  // value = m * 2^(e - 134); in Q16: m << (e - 118)
    int32_t v = shift >= 0 ? (shift > 22 ? (m << 22) : (m << shift)) : (shift < -31 ? 0 : (m >> -shift));
    return (b & 0x8000) ? -v : v;
}

// 2^frac(t) in Q16 ([65536, 131072)) and floor(t) for t = d * log2(e), d <= 0 in Q16: exp(d) = p * 2^(n - 16).
inline uint32_t exp2_parts(int32_t d, int32_t& n) {
    const int32_t t = static_cast<int32_t>((static_cast<int64_t>(d) * 94548) >> 16);  // d * log2(e), Q16
    n = t >> 16;                                                                      // floor
    const uint32_t f = static_cast<uint32_t>(t) & 0xFFFF;                             // [0, 1) in Q16
    uint32_t p = 87;                                                                  // 2^f, Horner, Q16
    p = 630 + ((p * f) >> 16);
    p = 3638 + ((p * f) >> 16);
    p = 15743 + ((p * f) >> 16);
    p = 45426 + ((p * f) >> 16);
    p = 65536 + ((p * f) >> 16);
    return p;
}

// Unsigned fixed point with `frac` fractional bits, value in [0, 1] -> BF16 bits, round to nearest.
inline uint16_t fixed_to_bf16(uint64_t w, int32_t frac) {
    if (w == 0) {
        return 0;
    }
    const uint32_t hi = static_cast<uint32_t>(w >> 32);
    int32_t msb = hi ? 63 - __builtin_clz(hi) : 31 - __builtin_clz(static_cast<uint32_t>(w));
    const int32_t shift = msb - 7;
    uint64_t m = shift > 0 ? (w + (1ull << (shift - 1))) >> shift : w << -shift;
    if (m >= 0x100) {  // rounding carried into a new bit
        m >>= 1;
        msb += 1;
    }
    return static_cast<uint16_t>(((127 + msb - frac) << 7) | (static_cast<uint32_t>(m) & 0x7F));
}

inline uint32_t order_key(uint32_t b) { return (b & 0x8000) ? (~b & 0xFFFF) : (b | 0x8000); }

// Top-k of the n BF16 logits (sorted descending, ties to the lower id) -> ids[k], softmax weights[k] (BF16 bits).
template <uint32_t topk, uint32_t n>
inline void topk_softmax(volatile tt_l1_ptr uint16_t* logits, uint32_t* ids, uint16_t* weights) {
    uint32_t keys[topk];
    for (uint32_t j = 0; j < topk; ++j) {
        keys[j] = 0;
        ids[j] = 0;
    }
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t key = order_key(logits[i]);
        if (key <= keys[topk - 1]) {
            continue;
        }
        uint32_t j = topk - 1;
        while (j > 0 && key > keys[j - 1]) {
            keys[j] = keys[j - 1];
            ids[j] = ids[j - 1];
            --j;
        }
        keys[j] = key;
        ids[j] = i;
    }
    // softmax over the selected: e_j = exp(v_j - v_0) = p_j 2^(n_j - 16) in Q32 (p_j << (16 + n_j), n_j <= 0;
    // terms below 2^-16 of the largest dropped), weights in Q27 from a 64-bit division (~1e-5 relative)
    const int32_t v0 = bf16_to_q16(logits[ids[0]]);
    uint64_t e[topk];
    uint64_t sum = 0;
    for (uint32_t j = 0; j < topk; ++j) {
        int32_t nj;
        const uint32_t p = exp2_parts(bf16_to_q16(logits[ids[j]]) - v0, nj);
        e[j] = nj < -16 ? 0 : static_cast<uint64_t>(p) << (16 + nj);
        sum += e[j];
    }
    for (uint32_t j = 0; j < topk; ++j) {
        weights[j] = fixed_to_bf16((e[j] << 27) / sum, 27);
    }
}
}  // namespace moe_topk
