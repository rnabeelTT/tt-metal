// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Block attention residual mix (tt/mixw.py MixKernel.multi), reader for one row tile (T <= 32): the n sources and
// their logits come from separate tensors (no stacked copy): source i's H tiles from S_i, its nq logit tiles from
// tiles off_i .. off_i + nq - 1 of LG_i (a block's logits against every later query, or the current source's own).
// Same CB layout as mixw_reader.cpp (logit tile i * nq + q; source tiles ordered [t][i]).
//
// runtime args: t0, nt, then per source i: s_addr, lg_addr, lg_off

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

    constexpr uint32_t cb_lg = get_compile_time_arg_val(0);
    constexpr uint32_t cb_s = get_compile_time_arg_val(1);
    constexpr uint32_t n = get_compile_time_arg_val(2);
    constexpr uint32_t nq = get_compile_time_arg_val(3);
    constexpr uint32_t F32 = 4096;
    constexpr auto s_args = TensorAccessorArgs<4>();
    constexpr auto lg_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();

    cb_reserve_back(cb_lg, n * nq);
    const uint32_t l = get_write_ptr(cb_lg);
    for (uint32_t i = 0; i < n; ++i) {
        const auto alg = TensorAccessor(lg_args, get_arg_val<uint32_t>(2 + 3 * i + 1), F32);
        const uint32_t off = get_arg_val<uint32_t>(2 + 3 * i + 2);
        for (uint32_t q = 0; q < nq; ++q) {
            read_tile(alg.get_noc_addr(off + q), l + (i * nq + q) * CT);
        }
    }
    cb_reserve_back(cb_s, n * nt);
    const uint32_t sp = get_write_ptr(cb_s);
    for (uint32_t i = 0; i < n; ++i) {
        const auto as = TensorAccessor(s_args, get_arg_val<uint32_t>(2 + 3 * i), F32);
        for (uint32_t t = 0; t < nt; ++t) {
            read_tile(as.get_noc_addr(t0 + t), sp + (t * n + i) * CT);
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_lg, n * nq);
    cb_push_back(cb_s, n * nt);
}
