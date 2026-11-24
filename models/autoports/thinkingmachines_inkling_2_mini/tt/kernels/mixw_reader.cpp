// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Block attention residual mix (tt/mixw.py), reader: per row tile, every logit tile (n sources x nq queries; each
// query's logit in every column of its tile), then this core's H tiles of the n sources (fp32), column by column.
// Source i: its own tensors (or slices of stacked ones) at (s_addr_i, s_off_i) / (lg_addr_i, lg_off_i) in tiles;
// row tile r, column t at s_off_i + r * Ht + t, query q's logits at lg_off_i + r * lg_stride_i + q (a block's logits
// against every later query: this mix's queries at a column offset of a wider tensor).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// MIX_TH < 32 (decode, MIX_T <= 16 live rows): the CB tiles are MIX_TH x 32 (faces MIX_TH x 16): rows 0 .. MIX_T - 1
// of faces 0 / 1 move between them and the tensors' 32 x 32 tiles
#ifndef MIX_TH
#define MIX_TH 32
#define MIX_T 32
#endif
constexpr uint32_t CT = MIX_TH * 128;  // a CB tile (fp32)
inline void read_tile(uint64_t src, uint32_t dst) {
    if constexpr (MIX_TH == 32) {
        noc_async_read(src, dst, 4096);
    } else {
        noc_async_read(src, dst, MIX_T * 64);
        noc_async_read(src + 1024, dst + MIX_TH * 64, MIX_T * 64);
    }
}

void kernel_main() {
    const uint32_t t0 = get_arg_val<uint32_t>(0);
    const uint32_t nt = get_arg_val<uint32_t>(1);
    const uint32_t r0 = get_arg_val<uint32_t>(2);  // this core's row tiles r0 .. r0 + nr - 1
    const uint32_t nr = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_lg = get_compile_time_arg_val(0);
    constexpr uint32_t cb_s = get_compile_time_arg_val(1);
    constexpr uint32_t n = get_compile_time_arg_val(2);
    constexpr uint32_t nq = get_compile_time_arg_val(3);
    constexpr uint32_t Ht = get_compile_time_arg_val(4);
    constexpr uint32_t F32 = 4096;
    constexpr auto s_args = TensorAccessorArgs<5>();
    constexpr auto lg_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    // per source: s_addr, s_off, lg_addr, lg_off, lg_stride
    auto src_arg = [](uint32_t i, uint32_t k) { return get_arg_val<uint32_t>(4 + 5 * i + k); };

    for (uint32_t r = r0; r < r0 + nr; ++r) {
        cb_reserve_back(cb_lg, n * nq);
        const uint32_t l = get_write_ptr(cb_lg);
        for (uint32_t i = 0; i < n; ++i) {
            const auto alg = TensorAccessor(lg_args, src_arg(i, 2), F32);
            const uint32_t base = src_arg(i, 3) + r * src_arg(i, 4);
            for (uint32_t q = 0; q < nq; ++q) {
                read_tile(alg.get_noc_addr(base + q), l + (i * nq + q) * CT);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_lg, n * nq);
        for (uint32_t t = 0; t < nt; ++t) {
            cb_reserve_back(cb_s, n);
            const uint32_t sp = get_write_ptr(cb_s);
            for (uint32_t i = 0; i < n; ++i) {
                const auto as = TensorAccessor(s_args, src_arg(i, 0), F32);
                read_tile(as.get_noc_addr(src_arg(i, 1) + r * Ht + t0 + t), sp + i * CT);
            }
            noc_async_read_barrier();
            cb_push_back(cb_s, n);
        }
    }
}
