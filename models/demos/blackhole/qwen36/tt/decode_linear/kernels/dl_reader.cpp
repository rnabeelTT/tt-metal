// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode linear, weight streamer (NCRISC, NOC0) (decode_linear/ops.py DecodeLinear): this core's `ncols` output columns
// of DRAM bank `bank` are one contiguous run of ncols x kt weight tiles (the per-bank column-major decode copy),
// streamed column by column into cb_w in `page_bytes` NOC packets, transaction-id pipelined with nbuf - 1 columns in
// flight (the moe_decode gate|up reader's scheme).
//
// runtime args: [w_addr, bank, col0 (column within the bank), ncols, vc]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t bank = get_arg_val<uint32_t>(1);
    const uint32_t col0 = get_arg_val<uint32_t>(2);
    const uint32_t ncols = get_arg_val<uint32_t>(3);
    const uint32_t vc = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t kt = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t num_buffers = get_compile_time_arg_val(4);
    constexpr uint32_t col_bytes = kt * tile_bytes;
    constexpr uint32_t pages = col_bytes / page_bytes;
    static_assert(pages * page_bytes == col_bytes, "packets must tile a column");
    static_assert(num_buffers >= 2, "ring of columns");

    if (ncols == 0) {
        return;
    }
    const uint64_t bank_noc = get_noc_addr_from_bank_id<true>(bank, w_addr);
    noc_async_read_one_packet_set_state<true>(bank_noc, page_bytes, vc);
    constexpr uint32_t extra_in_flight = num_buffers - 2;
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    cb_reserve_back(cb_w, kt * (extra_in_flight + 1));
    uint32_t l1 = get_write_ptr(cb_w);
    const uint32_t cb_base = l1;
    const uint32_t cb_end = cb_base + num_buffers * col_bytes;
    uint32_t free_blocks = num_buffers;
    uint32_t trid = 1;
    uint32_t trid_to_wait = 1;
    uint32_t src = col0 * col_bytes;
    for (uint32_t c = 0; c < ncols; ++c) {
        noc_async_read_set_trid(trid);
        for (uint32_t p = 0; p < pages; ++p) {
            noc_async_read_one_packet_with_state_with_trid((uint32_t)bank_noc, src, l1, trid);
            src += page_bytes;
            l1 += page_bytes;
        }
        if (free_blocks == num_buffers - extra_in_flight) {
            noc_async_read_barrier_with_trid(trid_to_wait);
            cb_push_back(cb_w, kt);
            trid_to_wait = trid_to_wait == num_buffers ? 1 : trid_to_wait + 1;
            cb_reserve_back(cb_w, kt * (extra_in_flight + 1));
        } else {
            free_blocks -= 1;
        }
        trid = trid == num_buffers ? 1 : trid + 1;
        if (l1 >= cb_end) {
            l1 = cb_base;
        }
    }
    const uint32_t pending = num_buffers - free_blocks;
    for (uint32_t i = 0; i < pending; ++i) {
        noc_async_read_barrier_with_trid(trid_to_wait);
        cb_push_back(cb_w, kt);
        trid_to_wait = trid_to_wait == num_buffers ? 1 : trid_to_wait + 1;
    }
    noc_async_read_one_packet_set_state<true>(get_noc_addr(MEM_ZEROS_BASE), 0, 1);  // restore the read cmd buffer
    noc_async_read_set_trid(0);
}
