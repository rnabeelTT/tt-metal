// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The sampling tail's top-k from the LM head's per-core lists (tt/sampling.py HeadTopK; head_writer.cpp HEAD_TOPK).
// LH_DMAX: one core; row r's shift bf16(max) of the group's rows (the lists' heads) as fp32 into column 0 of row r of
//   the [1, 1, rows, 1] output tile (the tail's die-max exchange input, as RowMax writes it).
// else: one core per row r of the group: the row's top 32 by (bf16(x - m), then the lower column), m = bf16(row max)
//   or (LH_MT) column 0 of row r of the shift tile; written into row r of the [32, 32] value (bf16) / index (IB bytes)
//   tiles as RowsTopK writes them (row 0's core also fills rows >= ROWS with its row).
//
// compile args: cb, NC, IB, ROWS, accessors (lists, logits, values, indices, shift)
// runtime args: lists, logits, values, indices, shift addresses, head row, row r; per head core: first tile, tiles

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "lhsel.hpp"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t NC = get_compile_time_arg_val(1);
    constexpr uint32_t IB = get_compile_time_arg_val(2);
    constexpr uint32_t ROWS = get_compile_time_arg_val(3);
    constexpr auto al = TensorAccessorArgs<4>();
    constexpr auto ag = TensorAccessorArgs<al.next_compile_time_args_offset()>();
    constexpr auto av = TensorAccessorArgs<ag.next_compile_time_args_offset()>();
    constexpr auto ai = TensorAccessorArgs<av.next_compile_time_args_offset()>();
    constexpr auto am = TensorAccessorArgs<ai.next_compile_time_args_offset()>();
    constexpr uint32_t LP = LH_PAGE;
    constexpr uint32_t RB = 32;  // rescan tiles per batch (two 64 B face rows each)

    const uint32_t l_addr = get_arg_val<uint32_t>(0);
    const uint32_t g_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t i_addr = get_arg_val<uint32_t>(3);
    const uint32_t m_addr = get_arg_val<uint32_t>(4);
    const uint32_t hrow = get_arg_val<uint32_t>(5);
    const uint32_t r = get_arg_val<uint32_t>(6);
    uint32_t t0[NC], nt[NC];
    for (uint32_t c = 0; c < NC; ++c) {
        t0[c] = get_arg_val<uint32_t>(7 + 2 * c);
        nt[c] = get_arg_val<uint32_t>(8 + 2 * c);
    }
    const auto s_l = TensorAccessor(al, l_addr, LP);
    const uint32_t buf = get_write_ptr(cb);

#if LH_DMAX
    // scratch: NC x ROWS list heads (64 B each), then the output tile
    for (uint32_t q = 0; q < ROWS; ++q) {
        for (uint32_t c = 0; c < NC; ++c) {
            if (nt[c]) {
                noc_async_read(s_l.get_noc_addr(c * 32 + hrow + q), buf + (q * NC + c) * 64, 64);
            }
        }
    }
    const uint32_t ot = buf + ROWS * NC * 64;
    volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ot);
    for (uint32_t i = 0; i < 1024; ++i) {
        o[i] = 0;
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    for (uint32_t q = 0; q < ROWS; ++q) {
        int32_t b = INT32_MIN;
        for (uint32_t c = 0; c < NC; ++c) {
            const int32_t s = *reinterpret_cast<volatile tt_l1_ptr int32_t*>(buf + (q * NC + c) * 64);
            if (nt[c] && s > b) {
                b = s;
            }
        }
        const uint32_t u = lh_unsord(b);
        o[(q >> 4) * 512 + (q & 15) * 16] = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
    }
    const auto s_m = TensorAccessor(am, m_addr, 4096);
    noc_async_write(ot, s_m.get_noc_addr(0), 4096);
    noc_async_write_barrier();
#else
    // scratch: NC list pages, the shift tile's row (64 B), the output rows (values 64 B, indices 128 B), rescan staging,
    // the best 32 and the merge pointers
    const uint32_t mrow = buf + NC * LP, vrow = mrow + 64, irow = vrow + 64, rs = irow + 128, bw = rs + RB * 128;
    for (uint32_t c = 0; c < NC; ++c) {
        if (nt[c]) {
            noc_async_read(s_l.get_noc_addr(c * 32 + hrow), buf + c * LP, LP);
        } else {
            reinterpret_cast<uint32_t*>(buf + c * LP)[2 * LH_KL] = 0;
            reinterpret_cast<uint32_t*>(buf + c * LP)[2 * LH_KL + 2] = 0;
        }
    }
#if LH_MT
    const auto s_m = TensorAccessor(am, m_addr, 4096);
    noc_async_read(s_m.get_noc_addr(0) + (r >> 4) * 2048 + (r & 15) * 64, mrow, 64);
#endif
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t* pages = reinterpret_cast<const uint32_t*>(buf);
#if LH_MT
    const uint32_t mu = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mrow);
#else
    int32_t top = INT32_MIN;  // (each list holds its core's best first)
    for (uint32_t c = 0; c < NC; ++c) {
        if (pages[c * (LP / 4) + 2 * LH_KL] && static_cast<int32_t>(pages[c * (LP / 4)]) > top) {
            top = static_cast<int32_t>(pages[c * (LP / 4)]);
        }
    }
    uint32_t mu = lh_unsord(top);
    mu = (mu + 0x7FFFu + ((mu >> 16) & 1u)) & 0xFFFF0000u;
#endif
    const auto s_g = TensorAccessor(ag, g_addr, 4096);
    const uint32_t foff = (hrow >> 4) * 2048 + (hrow & 15) * 64;  // the row in face 0 / 2 of a logits tile
    auto rescan = [&](uint32_t c, int32_t thr, LhBest& best) {  // the core's dropped values (<= thr) from the logits
        const int32_t lo = lh_xlo(INT32_MIN + 1, thr, mu, best.last());
        for (uint32_t b0 = 0; b0 < nt[c]; b0 += RB) {
            const uint32_t nb = nt[c] - b0 < RB ? nt[c] - b0 : RB;
            for (uint32_t k = 0; k < nb; ++k) {
                const uint64_t src = s_g.get_noc_addr(t0[c] + b0 + k) + foff;
                noc_async_read(src, rs + k * 128, 64);
                noc_async_read(src + 1024, rs + k * 128 + 64, 64);
            }
            noc_async_read_barrier();
            invalidate_l1_cache();
            const uint32_t* p = reinterpret_cast<const uint32_t*>(rs);
            for (uint32_t k = 0; k < nb * 32; ++k) {
                const int32_t s = lh_sord(p[k]);
                if (s >= lo && s <= thr) {
                    best.add(lh_ord16(lh_key(p[k], mu)), (t0[c] + b0 + k / 32) * 32 + (k & 31));
                }
            }
        }
    };
    LhBest& best = *reinterpret_cast<LhBest*>(bw);
    lh_select<NC>(pages, mu, best, reinterpret_cast<uint32_t*>(bw + sizeof(LhBest)), rescan);
    // row r of the tiles: face (r / 16) * 2 + j / 16, row r % 16
    uint16_t* vr = reinterpret_cast<uint16_t*>(vrow);
    for (uint32_t j = 0; j < LH_K; ++j) {
        vr[j] = lh_unord16(best.key[j]);
        if constexpr (IB == 4) {
            reinterpret_cast<uint32_t*>(irow)[j] = best.col[j];
        } else {
            reinterpret_cast<uint16_t*>(irow)[j] = static_cast<uint16_t>(best.col[j]);
        }
    }
    const auto s_v = TensorAccessor(av, v_addr, 2048);
    const auto s_i = TensorAccessor(ai, i_addr, IB * 1024);
    const uint64_t vt = s_v.get_noc_addr(0), it = s_i.get_noc_addr(0);
    auto put = [&](uint32_t q) {
        for (uint32_t f = 0; f < 2; ++f) {
            const uint32_t face = (q >> 4) * 2 + f, row = q & 15;
            noc_async_write(vrow + f * 32, vt + face * 512 + row * 32, 32);
            noc_async_write(irow + f * 16 * IB, it + face * 256 * IB + row * 16 * IB, 16 * IB);
        }
    };
    put(r);
    if (r == 0) {
        for (uint32_t q = ROWS; q < 32; ++q) {
            put(q);
        }
    }
    noc_async_write_barrier();
#endif
}
