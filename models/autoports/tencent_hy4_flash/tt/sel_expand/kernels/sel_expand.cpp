// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Decode selection expansion (both RISCs, one row per core; RISC `part` builds groups [g0, g1) of the row, part 1
// also copies the tail): see ../__init__.py.
// runtime args: [ids, base, tail, sel, idx, row, g0, g1, part]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t ids_addr = get_arg_val<uint32_t>(0);
    const uint32_t base_addr = get_arg_val<uint32_t>(1);
    const uint32_t tail_addr = get_arg_val<uint32_t>(2);
    const uint32_t sel_addr = get_arg_val<uint32_t>(3);
    const uint32_t idx_addr = get_arg_val<uint32_t>(4);
    const uint32_t row = get_arg_val<uint32_t>(5);
    const uint32_t g0 = get_arg_val<uint32_t>(6);
    const uint32_t g1 = get_arg_val<uint32_t>(7);
    const uint32_t part = get_arg_val<uint32_t>(8);

    constexpr uint32_t K = get_compile_time_arg_val(0);
    constexpr uint32_t NSEL = get_compile_time_arg_val(1);
    constexpr uint32_t RR = get_compile_time_arg_val(2);
    constexpr uint32_t TW = get_compile_time_arg_val(3);
    constexpr uint32_t R = get_compile_time_arg_val(4);
    constexpr auto ids_args = TensorAccessorArgs<5>();
    constexpr auto b_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto t_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<t_args.next_compile_time_args_offset()>();
    constexpr auto i_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr uint32_t NS = NSEL * RR;

    const uint32_t cb = part;  // CB 0 / 1: this RISC's scratch
    const uint32_t buf = get_write_ptr(cb);
    const uint32_t n = (g1 - g0) * RR;
    // ids row | this RISC's sel part (+ the tail words appended, part 1) | scratch (64 B aligned)
    const uint32_t ids_l1 = buf, out_l1 = buf + K * 4, tail_l1 = out_l1 + ((n + TW) * 4 + 63) / 64 * 64 + 64;
    const auto s_sel = TensorAccessor(s_args, sel_addr);
    const auto s_idx = TensorAccessor(i_args, idx_addr);
    volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_l1);

    if (row < R) {
        const auto s_ids = TensorAccessor(ids_args, ids_addr);
        const auto s_b = TensorAccessor(b_args, base_addr);
        noc_async_read(s_ids.get_noc_addr(row), ids_l1, K * 4);
        noc_async_read(s_b.get_noc_addr(0), tail_l1, 64);  // element (0, 0) of the base tile: the first word
        noc_async_read_barrier();
        // base: an integer-valued fp32 (< 2^24) -> uint32
        const uint32_t fb = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tail_l1);
        const int32_t e = static_cast<int32_t>((fb >> 23) & 0xFF) - 127;
        const uint32_t m = (fb & 0x7FFFFF) | 0x800000;
        const uint32_t base = (fb & 0x7FFFFFFF) == 0 ? 0u : (e >= 23 ? m << (e - 23) : (e < 0 ? 0u : m >> (23 - e)));
        volatile tt_l1_ptr uint32_t* id = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ids_l1);
        uint32_t k = 0;
        for (uint32_t gi = g0; gi < g1; ++gi) {
            const uint32_t v = id[gi] * RR + base;
            for (uint32_t j = 0; j < RR; ++j) {
                o[k++] = v + j;
            }
        }
    } else {
        for (uint32_t k = 0; k < n; ++k) {
            o[k] = 0xFFFFFFFFu;
        }
    }
    const uint32_t off = g0 * RR * 4;
    noc_async_write(out_l1, s_sel.get_noc_addr(row) + off, n * 4);
    uint32_t nw = n;
    if (part == 1) {  // the tail row appended to this part of the idx row (one write: the tail's offset is unaligned)
        const auto s_t = TensorAccessor(t_args, tail_addr);
        noc_async_read(s_t.get_noc_addr(row), tail_l1, TW * 4);
        noc_async_read_barrier();
        volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tail_l1);
        for (uint32_t k = 0; k < TW; ++k) {
            o[n + k] = t[k];
        }
        nw += TW;
    }
    noc_async_write(out_l1, s_idx.get_noc_addr(row) + off, nw * 4);
    noc_async_write_barrier();
}
