// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Prefill attention prep in the projections' layout (tt/aprep.py NAT): the units of one tile row of the chunk, in order:
// NAT_NG q head groups, NAT_NG r head groups, NAT_NKV k heads, NAT_NKV v heads, the sink. A head group is up to NAT_HG
// consecutive heads that stay inside one 16-row face of the output tiles (their rows are written as one run).
// Types: 0 q, 1 r, 2 k, 3 v, 4 sink; heads [h0, h1).

#pragma once

#include <stdint.h>

#ifndef NAT_NH
#define NAT_NH 1
#define NAT_NKV 1
#endif
#ifndef NAT_STK
#define NAT_STK 0  // q / rq / sink out in the SDPA's stacked tiles (aprep_nat_writer.cpp); q / r groups = kv heads
#endif
#ifndef NAT_HG
#define NAT_HG 2  // heads per q / r unit (even: two heads' rows share a 64-byte DRAM word, so one core writes both)
#endif

constexpr uint32_t NAT_G = NAT_NH / NAT_NKV;  // q heads per kv head

constexpr uint32_t nat_ng() {
    if (NAT_STK) {
        return NAT_NKV;
    }
    uint32_t n = 0;
    for (uint32_t h = 0; h < NAT_NH;) {
        uint32_t e = h + NAT_HG;
        const uint32_t f = (h / 16 + 1) * 16;
        e = e < f ? e : f;
        e = e < NAT_NH ? e : NAT_NH;
        h = e;
        ++n;
    }
    return n;
}
constexpr uint32_t NAT_NG = nat_ng();
constexpr uint32_t NAT_NU = 2 * NAT_NG + 2 * NAT_NKV + 1;

inline void nat_group(uint32_t g, uint32_t& h0, uint32_t& h1) {
    if constexpr (NAT_STK) {
        h0 = g * NAT_G, h1 = h0 + NAT_G;
        return;
    }
    uint32_t h = 0;
    for (uint32_t i = 0;; ++i) {
        uint32_t e = h + NAT_HG;
        const uint32_t f = (h / 16 + 1) * 16;
        e = e < f ? e : f;
        e = e < NAT_NH ? e : NAT_NH;
        if (i == g) {
            h0 = h, h1 = e;
            return;
        }
        h = e;
    }
}

inline void nat_unit(uint32_t t, uint32_t& type, uint32_t& h0, uint32_t& h1) {
    if (t < 2 * NAT_NG) {
        type = t / NAT_NG;
        nat_group(t % NAT_NG, h0, h1);
    } else if (t < 2 * NAT_NG + 2 * NAT_NKV) {
        const uint32_t i = t - 2 * NAT_NG;
        type = 2 + i / NAT_NKV, h0 = i % NAT_NKV, h1 = h0 + 1;
    } else {
        type = 4, h0 = 0, h1 = NAT_NH;
    }
}

// byte offset of row r's half (columns 0-15 / 16-31) in a 32 x 32 tile: fp32 (64 B face rows) / bf16 (32 B)
inline uint32_t nat_f32_row(uint32_t r, uint32_t half) { return ((r >> 4) * 2 + half) * 1024 + (r & 15) * 64; }
inline uint32_t nat_bf_row(uint32_t r, uint32_t half) { return ((r >> 4) * 2 + half) * 512 + (r & 15) * 32; }
