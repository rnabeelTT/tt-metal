// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// DSpark drafter: row `tok` of markov_w1 (bf16 [V, 256] ROW_MAJOR) -> row 0 of a [1, 1, 1, 256] bf16 TILE tensor
// (8 tiles; the other rows untouched). l1: 64-aligned scratch of 512 + 1024 bytes.
#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

template <typename AW, typename AO>
inline void w1_row(const AW& s_w, const AO& s_o, uint32_t tok, uint32_t l1) {
    noc_async_read(s_w.get_noc_addr(tok), l1, 512);
    noc_async_read_barrier();
    // tile t's row 0: 16 values in face 0 (tile offset 0), 16 in face 1 (offset 512); staged 64-aligned
    const uint32_t st = l1 + 512;
    volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(st);
    for (uint32_t t = 0; t < 8; ++t) {
        for (uint32_t w = 0; w < 8; ++w) {
            dst[t * 32 + w] = src[t * 16 + w];
            dst[t * 32 + 16 + w] = src[t * 16 + 8 + w];
        }
    }
    for (uint32_t t = 0; t < 8; ++t) {
        const uint64_t d = s_o.get_noc_addr(t);
        noc_async_write(st + t * 128, d, 32);
        noc_async_write(st + t * 128 + 64, d + 512, 32);
    }
    noc_async_write_barrier();
}
