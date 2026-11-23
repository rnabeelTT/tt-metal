// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Lane compaction (see ../__init__.py): RISC `part` builds columns [c0, c1) of row `row`.
// runtime args: [sc, lmeta, out, row, c0, c1]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t sc_addr = get_arg_val<uint32_t>(0);
    const uint32_t m_addr = get_arg_val<uint32_t>(1);
    const uint32_t o_addr = get_arg_val<uint32_t>(2);
    const uint32_t row = get_arg_val<uint32_t>(3);
    const uint32_t c0 = get_arg_val<uint32_t>(4);
    const uint32_t c1 = get_arg_val<uint32_t>(5);
    constexpr uint32_t WS = get_compile_time_arg_val(0);
    constexpr uint32_t WC = get_compile_time_arg_val(1);
    constexpr auto sc_args = TensorAccessorArgs<2>();
    constexpr auto m_args = TensorAccessorArgs<sc_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<m_args.next_compile_time_args_offset()>();
    constexpr uint16_t NEG_INF = 0xFF80;

    const uint32_t cb = get_arg_val<uint32_t>(4) == 0 ? 0 : 1;
    const uint32_t buf = get_write_ptr(cb), meta_l1 = buf + (c1 - c0) * 2 + 64;
    const auto s_m = TensorAccessor(m_args, m_addr);
    noc_async_read(s_m.get_noc_addr(row), meta_l1, 64);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_l1);
    const uint32_t lo = meta[0], vis = meta[3] ? meta[1] : 0;
    // copy [c0, min(c1, vis)) from sc[row, lo + c0 ..] (lo: a multiple of 32 groups: aligned), -inf after
    const uint32_t ce = vis < c0 ? c0 : (vis < c1 ? vis : c1);
    if (ce > c0) {
        const auto s_sc = TensorAccessor(sc_args, sc_addr);
        uint32_t nb = ((ce - c0) * 2 + 63) / 64 * 64;
        const uint32_t room = (WS - lo - c0) * 2;
        if (nb > room) {
            nb = room;
        }
        noc_async_read(s_sc.get_noc_addr(row) + (lo + c0) * 2, buf, nb);
        noc_async_read_barrier();
    }
    volatile tt_l1_ptr uint16_t* b = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf);
    uint32_t k = ce - c0;
    for (; k < c1 - c0 && (k & 1); ++k) {
        b[k] = NEG_INF;
    }
    volatile tt_l1_ptr uint32_t* b32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf);
    for (uint32_t w = k / 2; w < (c1 - c0) / 2; ++w) {
        b32[w] = (static_cast<uint32_t>(NEG_INF) << 16) | NEG_INF;
    }
    const auto s_o = TensorAccessor(o_args, o_addr);
    noc_async_write(buf, s_o.get_noc_addr(row) + c0 * 2, (c1 - c0) * 2);
    noc_async_write_barrier();
}
