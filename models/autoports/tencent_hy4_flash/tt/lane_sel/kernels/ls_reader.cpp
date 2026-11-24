// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Lane-aware score, reader (see ../__init__.py LaneSelOp.score): lmeta -> schedule; the head-weight columns w[:, h]
// as 16 bcast tiles (column 0, as indexer_score's reader expands them); per compute unit the sub-run's packed query
// tile (row 2 h + (r & 1) = q[h, r] for its rows r, 4 dim tiles; rebuilt when the sub-run changes) and the key tile's
// 4 dim tiles. common runtime args: [q, w, k, lmeta]; the core index comes from the logical coordinates.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ls_sched.hpp"

constexpr uint32_t cb_k = 0, cb_q = 1, cb_w = 2, cb_meta = 6, cb_scr = 7;
constexpr uint32_t TB = 2048;  // bf16 tile bytes

// byte offset of row r's half `half` (columns 16 half ..) in a bf16 tile
inline uint32_t row_off(uint32_t r, uint32_t half) { return ((r / 16) * 2 + half) * 512 + (r % 16) * 32; }

void kernel_main() {
    const uint32_t q_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t k_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t m_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t core = ls_core();
    constexpr uint32_t NC = get_compile_time_arg_val(0);
    constexpr auto q_args = TensorAccessorArgs<1>();
    constexpr auto w_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto k_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto m_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    const auto s_q = TensorAccessor(q_args, q_addr);
    const auto s_w = TensorAccessor(w_args, w_addr);
    const auto s_k = TensorAccessor(k_args, k_addr);
    const auto s_m = TensorAccessor(m_args, m_addr);

    // scratch: lmeta 2 KB | w page 2 KB
    const uint32_t scr = get_write_ptr(cb_scr);
    const uint32_t m_l1 = scr, w_l1 = scr + 2048;
    for (uint32_t r = 0; r < 32; ++r) {
        noc_async_read(s_m.get_noc_addr(r), m_l1 + r * 64, 64);
    }
    noc_async_read(s_w.get_noc_addr(0), w_l1, TB);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(m_l1);
    LsSched s;
    ls_build(meta, s);
    uint32_t b, e;
    ls_range(s.C, core, NC, b, e);

    // the compute descriptor: [count, new-sub-run flag per unit]
    cb_reserve_back(cb_meta, 1);
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta));
    d[0] = e - b;
    {
        LsCursor c = ls_seek(s, b, false);
        uint32_t prev = 0xFFFFFFFF;
        for (uint32_t u = b; u < e; ++u) {
            d[1 + u - b] = c.run != prev;
            prev = c.run;
            ls_next(s, c, false);
        }
    }
    cb_push_back(cb_meta, 1);
    if (e == b) {
        return;
    }

    // w [32 rows, 16 heads] -> tile h: column 0 = w[:, h] (word pairs, as indexer_score's expand_w_group)
    cb_reserve_back(cb_w, 16);
    {
        const uint32_t wb = get_write_ptr(cb_w);
        volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w_l1);
        for (uint32_t h = 0; h < 16; h += 2) {
            volatile tt_l1_ptr uint32_t* d0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wb + h * TB);
            volatile tt_l1_ptr uint32_t* d1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wb + (h + 1) * TB);
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t ro = (r / 16) * 2 * 128 + (r % 16) * 8;  // words
                const uint32_t pair = src[ro + h / 2];
                d0[ro] = pair;
                d1[ro] = pair >> 16;
            }
        }
    }
    cb_push_back(cb_w, 16);

    LsCursor c = ls_seek(s, b, false);
    uint32_t prev = 0xFFFFFFFF;
    for (uint32_t u = b; u < e; ++u) {
        if (c.run != prev) {
            prev = c.run;
            cb_reserve_back(cb_q, 4);
            const uint32_t qd = get_write_ptr(cb_q);
            // row r -> packed row 2 h + (r & 1): source and destination agree mod 64 B, so the rows are read straight
            // into place. A sub-run inside one aligned row pair (one row, or two from an even row) takes 64 B reads
            // (the pair's both rows; an unused slot is ignored), otherwise 32 B reads per row.
            const uint32_t r0 = s.r0[c.run], n = s.n[c.run];
            if (n == 1 || (r0 & 1) == 0) {
                const uint32_t ra = r0 & ~1u;
                for (uint32_t h = 0; h < 16; ++h) {
                    for (uint32_t dt = 0; dt < 4; ++dt) {
                        const uint64_t pg = s_q.get_noc_addr(h * 4 + dt);
                        for (uint32_t half = 0; half < 2; ++half) {
                            noc_async_read(pg + row_off(ra, half), qd + dt * TB + row_off(2 * h, half), 64);
                        }
                    }
                }
            } else {
                for (uint32_t r = r0; r < r0 + n; ++r) {
                    for (uint32_t h = 0; h < 16; ++h) {
                        const uint32_t p = 2 * h + (r & 1);
                        for (uint32_t dt = 0; dt < 4; ++dt) {
                            const uint64_t pg = s_q.get_noc_addr(h * 4 + dt);
                            for (uint32_t half = 0; half < 2; ++half) {
                                noc_async_read(pg + row_off(r, half), qd + dt * TB + row_off(p, half), 32);
                            }
                        }
                    }
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_q, 4);
        }
        cb_reserve_back(cb_k, 4);
        const uint32_t kd = get_write_ptr(cb_k);
        const uint32_t kt = s.lo[c.run] / 32 + c.kt;
        for (uint32_t dt = 0; dt < 4; ++dt) {
            noc_async_read(s_k.get_noc_addr(kt * 4 + dt), kd + dt * TB, TB);
        }
        noc_async_read_barrier();
        cb_push_back(cb_k, 4);
        ls_next(s, c, false);
    }
}
