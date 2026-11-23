// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Lane expansion (see ../__init__.py): RISC `part` builds groups [g0, g1) of a selecting row (part 1 appends the
// tail); a dense row is copied by part 0.
// runtime args: [ids, lmeta, tail, dense, idx, row, g0, g1, part]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t ids_addr = get_arg_val<uint32_t>(0);
    const uint32_t m_addr = get_arg_val<uint32_t>(1);
    const uint32_t t_addr = get_arg_val<uint32_t>(2);
    const uint32_t d_addr = get_arg_val<uint32_t>(3);
    const uint32_t x_addr = get_arg_val<uint32_t>(4);
    const uint32_t row = get_arg_val<uint32_t>(5);
    const uint32_t g0 = get_arg_val<uint32_t>(6);
    const uint32_t g1 = get_arg_val<uint32_t>(7);
    const uint32_t part = get_arg_val<uint32_t>(8);
    constexpr uint32_t K = get_compile_time_arg_val(0);
    constexpr uint32_t NSEL = get_compile_time_arg_val(1);
    constexpr uint32_t RR = get_compile_time_arg_val(2);
    constexpr uint32_t TW = get_compile_time_arg_val(3);
    constexpr uint32_t KD = get_compile_time_arg_val(4);
    constexpr auto ids_args = TensorAccessorArgs<5>();
    constexpr auto m_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto t_args = TensorAccessorArgs<m_args.next_compile_time_args_offset()>();
    constexpr auto d_args = TensorAccessorArgs<t_args.next_compile_time_args_offset()>();
    constexpr auto x_args = TensorAccessorArgs<d_args.next_compile_time_args_offset()>();

    const uint32_t buf = get_write_ptr(part);
    const uint32_t meta_l1 = buf, ids_l1 = buf + 64, out_l1 = ids_l1 + K * 4;
    const auto s_m = TensorAccessor(m_args, m_addr);
    const auto s_x = TensorAccessor(x_args, x_addr);
    noc_async_read(s_m.get_noc_addr(row), meta_l1, 64);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_l1);
    if (!meta[3]) {  // dense row (or padding): its host list
        if (part == 0) {
            const auto s_d = TensorAccessor(d_args, d_addr);
            noc_async_read(s_d.get_noc_addr(row), out_l1, KD * 4);
            noc_async_read_barrier();
            noc_async_write(out_l1, s_x.get_noc_addr(row), KD * 4);
            noc_async_write_barrier();
        }
        return;
    }
    const uint32_t base = meta[2];
    const auto s_ids = TensorAccessor(ids_args, ids_addr);
    noc_async_read(s_ids.get_noc_addr(row), ids_l1, K * 4);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* id = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ids_l1);
    volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_l1);
    uint32_t k = 0;
    for (uint32_t gi = g0; gi < g1; ++gi) {
        const uint32_t v = id[gi] * RR + base;
        for (uint32_t j = 0; j < RR; ++j) {
            o[k++] = v + j;
        }
    }
    if (part == 1) {  // + the tail (its offset in the row is unaligned: appended here, one write)
        const uint32_t t_l1 = out_l1 + ((k + TW) * 4 + 63) / 64 * 64 + 64;
        const auto s_t = TensorAccessor(t_args, t_addr);
        noc_async_read(s_t.get_noc_addr(row), t_l1, TW * 4);
        noc_async_read_barrier();
        volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(t_l1);
        for (uint32_t i = 0; i < TW; ++i) {
            o[k + i] = t[i];
        }
        k += TW;
    }
    noc_async_write(out_l1, s_x.get_noc_addr(row) + g0 * RR * 4, k * 4);
    noc_async_write_barrier();
}
