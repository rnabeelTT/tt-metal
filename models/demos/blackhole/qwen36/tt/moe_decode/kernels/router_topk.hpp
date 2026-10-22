// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Router top-k + softmax on a data-movement core (no FPU), for the Qwen3.5-MoE decode router
// (moe_decode/topk.py). Top-k by order-preserving integer keys of the BF16 logits (ties to the lower id, one pass
// of insertion into a sorted k-list, 32-bit word loads), softmax over the selected logits in 64-bit fixed point (exp2 = integer shift x
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

// sigmoid(x) of a BF16 logit -> BF16 bits: e = exp(-|x|) in Q32, 1 / (1 + e) or e / (1 + e) in Q27.
inline uint16_t sigmoid_bf16(uint16_t b) {
    const int32_t v = bf16_to_q16(b);
    int32_t n;
    const uint32_t p = exp2_parts(v < 0 ? v : -v, n);
    const uint64_t e = n < -32 ? 0 : (n < -16 ? static_cast<uint64_t>(p >> (-16 - n)) : static_cast<uint64_t>(p) << (16 + n));
    const uint64_t den = (1ull << 32) + e;
    return fixed_to_bf16(((v < 0 ? e : (1ull << 32)) << 27) / den, 27);
}

// Order-preserving integer key of BF16 bits (larger value -> larger key), branchless.
inline uint32_t order_key(uint32_t b) { return b ^ 0x8000 ^ ((0u - (b >> 15)) & 0x7FFF); }

template <uint32_t topk>
inline void insert(uint32_t key, uint32_t id, uint32_t* keys, uint32_t* ids) {
    uint32_t j = topk - 1;
    while (j > 0 && key > keys[j - 1]) {
        keys[j] = keys[j - 1];
        ids[j] = ids[j - 1];
        --j;
    }
    keys[j] = key;
    ids[j] = id;
}

// Top-k (sorted descending, ties to the lower id) of the BF16 values in 32-bit words [w0, w1) (value ids 2 w0 ..):
// keys[k] (order keys, 0 = empty) and ids[k].
template <uint32_t topk>
inline void select_range(const uint32_t* words, uint32_t w0, uint32_t w1, uint32_t* keys_out, uint32_t* ids_out) {
    // keys / ids live in registers (fully unrolled constant indexing; the kernel stack is in L1); a new value
    // bubbles up from the last slot with compare-swaps. The running k-th largest key is keys[topk - 1].
    uint32_t keys[topk], ids[topk];
#pragma GCC unroll 16
    for (uint32_t j = 0; j < topk; ++j) {
        keys[j] = 0;
        ids[j] = 0;
    }
    auto push = [&](uint32_t key, uint32_t id) __attribute__((always_inline)) {
        keys[topk - 1] = key;
        ids[topk - 1] = id;
#pragma GCC unroll 16
        for (uint32_t j = topk - 1; j > 0; --j) {
            if (keys[j] > keys[j - 1]) {
                const uint32_t tk = keys[j], ti = ids[j];
                keys[j] = keys[j - 1];
                ids[j] = ids[j - 1];
                keys[j - 1] = tk;
                ids[j - 1] = ti;
            }
        }
    };
#pragma GCC unroll 1
    for (uint32_t i = w0; i < w1; ++i) {
        const uint32_t v = words[i];
        // both halves' order keys at once: flip the sign bit, and the magnitude bits of negative values
        const uint32_t sgn = (v >> 15) & 0x00010001u;
        const uint32_t kk = v ^ 0x80008000u ^ ((sgn << 15) - sgn);
        const uint32_t k0 = kk & 0xFFFF, k1 = kk >> 16;
        if (k0 > keys[topk - 1]) {
            push(k0, 2 * i);
        }
        if (k1 > keys[topk - 1]) {
            push(k1, 2 * i + 1);
        }
    }
#pragma GCC unroll 16
    for (uint32_t j = 0; j < topk; ++j) {
        keys_out[j] = keys[j];
        ids_out[j] = ids[j];
    }
}

// keys / ids <- the top-k of (keys, ids) and (keys2, ids2), both sorted; on equal keys the first list (lower ids) wins.
template <uint32_t topk>
inline void merge(uint32_t* keys, uint32_t* ids, const uint32_t* keys2, const uint32_t* ids2) {
    uint32_t mk[topk], mi[topk];
    uint32_t a = 0, b = 0;
    for (uint32_t j = 0; j < topk; ++j) {
        if (keys[a] >= keys2[b]) {
            mk[j] = keys[a];
            mi[j] = ids[a++];
        } else {
            mk[j] = keys2[b];
            mi[j] = ids2[b++];
        }
    }
    for (uint32_t j = 0; j < topk; ++j) {
        keys[j] = mk[j];
        ids[j] = mi[j];
    }
}

template <uint32_t topk, uint32_t n>
inline void topk_select(const uint32_t* words, uint32_t* ids) {
    uint32_t keys[topk];
    select_range<topk>(words, 0, n / 2, keys, ids);
}

template <uint32_t topk>
inline void softmax_selected(const uint32_t* words, const uint32_t* ids, uint16_t* weights) {
    // softmax over the selected: e_j = exp(v_j - v_0) = p_j 2^(n_j - 16) in Q32 (p_j << (16 + n_j), n_j <= 0;
    // terms below 2^-16 of the largest dropped); weights in Q27 = e_j * (2^59 / sum) >> 32 (one 64-bit division)
    const uint16_t* vals = reinterpret_cast<const uint16_t*>(words);
    const int32_t v0 = bf16_to_q16(vals[ids[0]]);
    uint64_t e[topk];
    uint64_t sum = 0;
    for (uint32_t j = 0; j < topk; ++j) {
        int32_t nj;
        const uint32_t p = exp2_parts(bf16_to_q16(vals[ids[j]]) - v0, nj);
        e[j] = nj < -16 ? 0 : static_cast<uint64_t>(p) << (16 + nj);
        sum += e[j];
    }
    const uint64_t inv = (1ull << 59) / sum;  // sum in [2^32, 2^36)
    for (uint32_t j = 0; j < topk; ++j) {
        weights[j] = fixed_to_bf16((e[j] * inv) >> 32, 27);
    }
}

// Top-k of the n BF16 logits (sorted descending, ties to the lower id) -> ids[k], softmax weights[k] (BF16 bits).
// words: the logits as n / 2 32-bit words, read through the cache (the caller invalidates it after they arrive).
template <uint32_t topk, uint32_t n>
inline void topk_softmax(const uint32_t* words, uint32_t* ids, uint16_t* weights) {
    topk_select<topk, n>(words, ids);
    softmax_selected<topk>(words, ids, weights);
}
// Union of the routed experts of M route records (route table rows: k ids at [0, k), BF16 shared weight at 16,
// routed weights at [17, 17 + k)): uni[0 .. U) ascending; returns U (union slot 0 is the shared expert, slot 1 + i
// union expert i).
template <uint32_t topk>
inline uint32_t route_union(const uint16_t* table, uint32_t rows, uint16_t* uni) {
    uint32_t mask[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t j = 0; j < topk; ++j) {
            const uint32_t e = table[32 * r + j];
            mask[e >> 5] |= 1u << (e & 31);
        }
    }
    uint32_t u = 0;
    for (uint32_t w = 0; w < 8; ++w) {
        for (uint32_t m = mask[w]; m; m &= m - 1) {
            const uint32_t e = 32 * w + __builtin_ctz(m);
            uni[u] = static_cast<uint16_t>(e);
            ++u;
        }
    }
    return u;
}
}  // namespace moe_topk
