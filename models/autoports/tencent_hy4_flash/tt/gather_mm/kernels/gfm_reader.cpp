// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Hy4 prefill gather-matmul (tt/gather_mm), in0 reader: for each REAL group g of this core's [g0, g0 + n) (group id <
// num_experts; padding groups are skipped exactly as the in1 writer and the compute kernel skip them), the 32 source
// rows src_tok[32 g .. 32 g + 31] of the ROW-MAJOR expert input x [T, K] (one whole row per read, as moe_gather_rows)
// into cb_in0_rm; the compute kernel tilizes them (gfm_compute.cpp, IN0_TILIZE).
//   runtime args: [x_addr, src_addr, gid_addr, g0, n]
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t src_addr = get_arg_val<uint32_t>(1);
    const uint32_t gid_addr = get_arg_val<uint32_t>(2);
    const uint32_t g0 = get_arg_val<uint32_t>(3);
    const uint32_t n = get_arg_val<uint32_t>(4);

    constexpr uint32_t num_experts = get_compile_time_arg_val(0);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(1);  // K * 2
    constexpr uint32_t kt = get_compile_time_arg_val(2);         // K / 32: cb_in0_rm pages per group
    constexpr uint32_t gid_bytes = get_compile_time_arg_val(3);  // the group id stick's aligned page
    constexpr uint32_t src_page = get_compile_time_arg_val(4);   // the source-row stick's page bytes
    constexpr uint32_t cb_rm = get_compile_time_arg_val(5);
    constexpr uint32_t cb_src = get_compile_time_arg_val(6);
    constexpr uint32_t cb_gid = get_compile_time_arg_val(7);
    constexpr auto x_args = TensorAccessorArgs<8>();
    constexpr auto src_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto gid_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    const auto s_x = TensorAccessor(x_args, x_addr, row_bytes);
    const auto s_src = TensorAccessor(src_args, src_addr, src_page);
    const auto s_gid = TensorAccessor(gid_args, gid_addr, gid_bytes);
    const uint32_t gid_l1 = get_write_ptr(cb_gid);
    const uint32_t src_l1 = get_write_ptr(cb_src);
    noc_async_read(s_gid.get_noc_addr(0), gid_l1, gid_bytes);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint16_t* gid = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(gid_l1);

    volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_l1);
    for (uint32_t b = 0; b < n; ++b) {
        const uint32_t g = g0 + b;
        if (gid[g] >= num_experts) {
            continue;
        }
        noc_async_read(s_src.get_noc_addr(0, g * 32 * 4), src_l1, 32 * 4);
        noc_async_read_barrier();
        cb_reserve_back(cb_rm, kt);
        const uint32_t base = get_write_ptr(cb_rm);
        for (uint32_t r = 0; r < 32; ++r) {
            noc_async_read(s_x.get_noc_addr(src[r]), base + r * row_bytes, row_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb_rm, kt);
    }
}
