// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode linear, cross-op weight prefetch (decode_linear/ops.py DecodeLinear.prefetch_parts): runs inside an EARLIER
// program (e.g. the fused hyper-connection) on this linear's reader cores, which that program leaves idle. Reads this
// core's first `ncols` weight columns (the same contiguous bank range dl_reader.cpp streams) to L1 at `l1_dst`, the
// address the linear's weight CB will get (its first CB: the CB base). The host program declares a guard CB over that
// range, so no L1 buffer lives there; the linear's reader then pushes those columns without reading them.
//
// runtime args: [w_addr, bank, col0, ncols, vc, l1_dst]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t bank = get_arg_val<uint32_t>(1);
    const uint32_t col0 = get_arg_val<uint32_t>(2);
    const uint32_t ncols = get_arg_val<uint32_t>(3);
    const uint32_t vc = get_arg_val<uint32_t>(4);
    const uint32_t l1_dst = get_arg_val<uint32_t>(5);

    constexpr uint32_t kt = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t cb_guard = get_compile_time_arg_val(3);
    constexpr uint32_t col_bytes = kt * tile_bytes;
    constexpr uint32_t pages = col_bytes / page_bytes;
    static_assert(pages * page_bytes == col_bytes, "packets must tile a column");

    if (ncols == 0 || get_write_ptr(cb_guard) != l1_dst) {  // the guard CB is not where the linear's CB will be
        return;
    }
    const uint64_t bank_noc = get_noc_addr_from_bank_id<true>(bank, w_addr);
    noc_async_read_one_packet_set_state<true>(bank_noc, page_bytes, vc);
    uint32_t src = (uint32_t)bank_noc + col0 * col_bytes;
    uint32_t l1 = l1_dst;
    for (uint32_t i = 0; i < ncols * pages; ++i) {
        noc_async_read_one_packet_with_state<true, true>(src, l1, vc);
        src += page_bytes;
        l1 += page_bytes;
    }
    noc_async_read_barrier();
    noc_async_read_one_packet_set_state<true>(get_noc_addr(MEM_ZEROS_BASE), 0, 1);  // restore the read cmd buffer
}
