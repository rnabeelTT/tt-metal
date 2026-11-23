// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Lane-aware score, writer (see ../__init__.py LaneSelOp.score). Per compute unit: P's rows (2 h + (r & 1)) -> T_h
// row r (16 tiles for the compute's gate-mul), then the unit's accumulator rows -> out[r, 32 kt ..] (bf16 row-major,
// LaneSelOp.compact's format: columns >= the row's visible groups and rows that do not select: -inf). Then the fill
// units: -inf over [vt, VT) of each sub-run's rows.
// common runtime args: [lmeta, out]; the core index comes from the logical coordinates.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ls_sched.hpp"

constexpr uint32_t cb_p = 3, cb_t = 4, cb_acc = 5, cb_scr = 8;
constexpr uint32_t TB = 2048;
constexpr uint16_t NEG_INF = 0xFF80;

inline uint32_t row_off(uint32_t r, uint32_t half) { return ((r / 16) * 2 + half) * 512 + (r % 16) * 32; }

void kernel_main() {
    const uint32_t m_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t o_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t core = ls_core();
    constexpr uint32_t NC = get_compile_time_arg_val(0);
    constexpr auto m_args = TensorAccessorArgs<1>();
    constexpr auto o_args = TensorAccessorArgs<m_args.next_compile_time_args_offset()>();
    const auto s_m = TensorAccessor(m_args, m_addr);
    const auto s_o = TensorAccessor(o_args, o_addr);

    // scratch: lmeta 2 KB | row buffers 32 x 64 B | -inf 64 B
    const uint32_t scr = get_write_ptr(cb_scr);
    const uint32_t m_l1 = scr, rb_l1 = scr + 2048, ni_l1 = scr + 4096;
    for (uint32_t r = 0; r < 32; ++r) {
        noc_async_read(s_m.get_noc_addr(r), m_l1 + r * 64, 64);
    }
    volatile tt_l1_ptr uint32_t* ni = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ni_l1);
    for (uint32_t i = 0; i < 16; ++i) {
        ni[i] = (static_cast<uint32_t>(NEG_INF) << 16) | NEG_INF;
    }
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(m_l1);
    LsSched s;
    ls_build(meta, s);

    uint32_t b, e;
    ls_range(s.C, core, NC, b, e);
    // per unit: scatter P(u + 1) (the compute matmuls it before the gate-mul of u), then write acc(u)
    LsCursor cs = ls_seek(s, b, false), c = cs;
    auto scatter = [&](const LsCursor& x) {
        const uint32_t r0 = s.r0[x.run], n = s.n[x.run];
        cb_wait_front(cb_p, 1);
        cb_reserve_back(cb_t, 16);
        const uint32_t pa = get_read_ptr(cb_p), ta = get_write_ptr(cb_t);
        for (uint32_t r = r0; r < r0 + n; ++r) {
            for (uint32_t h = 0; h < 16; ++h) {
                const uint32_t p = 2 * h + (r & 1);
                for (uint32_t half = 0; half < 2; ++half) {
                    const uint32_t* sp = reinterpret_cast<const uint32_t*>(pa + row_off(p, half));
                    uint32_t* dp = reinterpret_cast<uint32_t*>(ta + h * TB + row_off(r, half));
                    dp[0] = sp[0]; dp[1] = sp[1]; dp[2] = sp[2]; dp[3] = sp[3];
                    dp[4] = sp[4]; dp[5] = sp[5]; dp[6] = sp[6]; dp[7] = sp[7];
                }
            }
        }
        cb_push_back(cb_t, 16);
        cb_pop_front(cb_p, 1);
    };
    if (b < e) {
        scatter(cs);
        ls_next(s, cs, false);
    }
    for (uint32_t u = b; u < e; ++u) {
        if (u + 1 < e) {
            scatter(cs);
            ls_next(s, cs, false);
        }
        const uint32_t r0 = s.r0[c.run], n = s.n[c.run];
        // accumulator rows -> out
        cb_wait_front(cb_acc, 1);
        noc_async_writes_flushed();  // (the row buffers of the last unit)
        const uint32_t aa = get_read_ptr(cb_acc);
        for (uint32_t j = 0; j < n; ++j) {
            const uint32_t r = r0 + j;
            const uint32_t sel = meta[r * 16 + 3], vis = meta[r * 16 + 1];
            const uint32_t c0 = c.kt * 32;
            const uint32_t lim = !sel || vis <= c0 ? 0 : (vis - c0 >= 32 ? 32 : vis - c0);
            const uint32_t rb = rb_l1 + r * 64;
            if (lim == 0) {
                noc_async_write(ni_l1, s_o.get_noc_addr(r) + c0 * 2, 64);
                continue;
            }
            uint32_t* dp = reinterpret_cast<uint32_t*>(rb);
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t* sp = reinterpret_cast<const uint32_t*>(aa + row_off(r, half));
                for (uint32_t i = 0; i < 8; ++i) {
                    dp[half * 8 + i] = sp[i];
                }
            }
            volatile tt_l1_ptr uint16_t* d16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rb);
            for (uint32_t k = lim; k < 32; ++k) {
                d16[k] = NEG_INF;
            }
            noc_async_write(rb, s_o.get_noc_addr(r) + c0 * 2, 64);
        }
        cb_pop_front(cb_acc, 1);
        ls_next(s, c, false);
    }
    // fill units
    ls_range(s.F, core, NC, b, e);
    c = ls_seek(s, b, true);
    for (uint32_t u = b; u < e; ++u) {
        for (uint32_t j = 0; j < s.n[c.run]; ++j) {
            noc_async_write(ni_l1, s_o.get_noc_addr(s.r0[c.run] + j) + c.kt * 64, 64);
        }
        ls_next(s, c, true);
    }
    noc_async_write_barrier();
}
