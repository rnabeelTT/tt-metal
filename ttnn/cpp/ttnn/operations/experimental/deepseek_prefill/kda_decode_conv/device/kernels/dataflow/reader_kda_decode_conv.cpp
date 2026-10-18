// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Tile element (r, c) of a 32 x 32 tile in face order (4 faces of 16 x 16, row-major inside a face).
constexpr uint32_t elem(uint32_t r, uint32_t c) { return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15); }

// Reads, per tile column c of this core: the current row tile (proj page c), the three history tiles and the four fp32
// tap tiles (row 0 copied down to rows 1 .. rows-1, so the compute kernel multiplies element-wise); once: the slot mask
// tile with column 0 copied across every column.
void kernel_main() {
    const uint32_t proj_addr = get_arg_val<uint32_t>(0);
    const uint32_t h0_addr = get_arg_val<uint32_t>(1);
    const uint32_t h1_addr = get_arg_val<uint32_t>(2);
    const uint32_t h2_addr = get_arg_val<uint32_t>(3);
    const uint32_t t0_addr = get_arg_val<uint32_t>(4);
    const uint32_t t1_addr = get_arg_val<uint32_t>(5);
    const uint32_t t2_addr = get_arg_val<uint32_t>(6);
    const uint32_t t3_addr = get_arg_val<uint32_t>(7);
    const uint32_t act_addr = get_arg_val<uint32_t>(8);
    const uint32_t start = get_arg_val<uint32_t>(9);
    const uint32_t n = get_arg_val<uint32_t>(10);
    const uint32_t rows = get_arg_val<uint32_t>(11);

    constexpr uint32_t cb_cur = get_compile_time_arg_val(0);
    constexpr uint32_t cb_h = get_compile_time_arg_val(1);
    constexpr uint32_t cb_tap = get_compile_time_arg_val(2);
    constexpr uint32_t cb_act = get_compile_time_arg_val(3);
    constexpr auto a_proj = TensorAccessorArgs<4>();
    constexpr auto a_h0 = TensorAccessorArgs<a_proj.next_compile_time_args_offset()>();
    constexpr auto a_h1 = TensorAccessorArgs<a_h0.next_compile_time_args_offset()>();
    constexpr auto a_h2 = TensorAccessorArgs<a_h1.next_compile_time_args_offset()>();
    constexpr auto a_t0 = TensorAccessorArgs<a_h2.next_compile_time_args_offset()>();
    constexpr auto a_t1 = TensorAccessorArgs<a_t0.next_compile_time_args_offset()>();
    constexpr auto a_t2 = TensorAccessorArgs<a_t1.next_compile_time_args_offset()>();
    constexpr auto a_t3 = TensorAccessorArgs<a_t2.next_compile_time_args_offset()>();
    constexpr auto a_act = TensorAccessorArgs<a_t3.next_compile_time_args_offset()>();

    const auto s_proj = TensorAccessor(a_proj, proj_addr);
    const auto s_h0 = TensorAccessor(a_h0, h0_addr);
    const auto s_h1 = TensorAccessor(a_h1, h1_addr);
    const auto s_h2 = TensorAccessor(a_h2, h2_addr);
    const auto s_t0 = TensorAccessor(a_t0, t0_addr);
    const auto s_t1 = TensorAccessor(a_t1, t1_addr);
    const auto s_t2 = TensorAccessor(a_t2, t2_addr);
    const auto s_t3 = TensorAccessor(a_t3, t3_addr);
    const auto s_act = TensorAccessor(a_act, act_addr);

    const uint32_t bf_page = get_local_cb_interface(cb_cur).fifo_page_size;
    const uint32_t fp_page = get_local_cb_interface(cb_tap).fifo_page_size;

    Noc noc;
    CircularBuffer cbc(cb_cur), cbh(cb_h), cbt(cb_tap), cba(cb_act);

    // slot mask: column 0 -> every column (rows 0 .. 31)
    cba.reserve_back(1);
    noc.async_read(s_act, cba, fp_page, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    {
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cba.get_write_ptr());
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t v = p[elem(r, 0)];
            for (uint32_t c = 1; c < 32; ++c) {
                p[elem(r, c)] = v;
            }
        }
    }
    cba.push_back(1);

    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t c = start + i;
        cbc.reserve_back(1);
        noc.async_read(s_proj, cbc, bf_page, {.page_id = c}, {.offset_bytes = 0});
        cbh.reserve_back(3);
        noc.async_read(s_h0, cbh, bf_page, {.page_id = c}, {.offset_bytes = 0});
        noc.async_read(s_h1, cbh, bf_page, {.page_id = c}, {.offset_bytes = bf_page});
        noc.async_read(s_h2, cbh, bf_page, {.page_id = c}, {.offset_bytes = 2 * bf_page});
        cbt.reserve_back(4);
        noc.async_read(s_t0, cbt, fp_page, {.page_id = c}, {.offset_bytes = 0});
        noc.async_read(s_t1, cbt, fp_page, {.page_id = c}, {.offset_bytes = fp_page});
        noc.async_read(s_t2, cbt, fp_page, {.page_id = c}, {.offset_bytes = 2 * fp_page});
        noc.async_read(s_t3, cbt, fp_page, {.page_id = c}, {.offset_bytes = 3 * fp_page});
        noc.async_read_barrier();
        // taps: row 0 -> rows 1 .. rows-1 (rows >= `rows` are padding, never read back as results)
        volatile tt_l1_ptr uint32_t* tp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbt.get_write_ptr());
        for (uint32_t k = 0; k < 4; ++k) {
            volatile tt_l1_ptr uint32_t* p = tp + k * (fp_page / 4);
            for (uint32_t r = 1; r < rows; ++r) {
                for (uint32_t col = 0; col < 32; ++col) {
                    p[elem(r, col)] = p[elem(0, col)];
                }
            }
        }
        cbc.push_back(1);
        cbh.push_back(3);
        cbt.push_back(4);
    }
}
