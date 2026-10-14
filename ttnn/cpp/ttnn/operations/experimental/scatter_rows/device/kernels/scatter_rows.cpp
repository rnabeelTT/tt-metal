// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
// Row scatter: for this core's rows r in [row_start, row_start + row_count): cache.page(idx[r]) = src.page(r),
// skipping idx == 0xFFFFFFFF. Pages are rows (ROW_MAJOR interleaved). Double-buffered: the write of row r overlaps
// the read of row r + 1.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t cache_addr = get_arg_val<uint32_t>(1);
    const uint32_t idx_addr = get_arg_val<uint32_t>(2);
    const uint32_t row_start = get_arg_val<uint32_t>(3);
    const uint32_t row_count = get_arg_val<uint32_t>(4);

    constexpr uint32_t row_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t cb_row = get_compile_time_arg_val(1);
    constexpr uint32_t cb_idx = get_compile_time_arg_val(2);
    constexpr uint32_t l1_align = get_compile_time_arg_val(3);
    constexpr uint32_t idx_stick_bytes = get_compile_time_arg_val(4);
    constexpr auto src_args = TensorAccessorArgs<5>();
    constexpr auto cache_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto idx_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();

    if (row_count == 0) {
        return;
    }
    Noc noc;
    const auto s_src = TensorAccessor(src_args, src_addr);
    const auto s_cache = TensorAccessor(cache_args, cache_addr);
    const auto s_idx = TensorAccessor(idx_args, idx_addr);

    // the whole index stick (a few KB) from offset 0: no source-alignment constraint on the slice start
    const uint32_t idx_l1 = DataflowBuffer(cb_idx).get_write_ptr();
    noc.async_read(s_idx, CoreLocalMem<uint32_t>(idx_l1), idx_stick_bytes, {.page_id = 0, .offset_bytes = 0}, {});
    noc.async_read_barrier();
    volatile tt_l1_ptr uint32_t* idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1) + row_start;

    DataflowBuffer dfb_row(cb_row);
    const uint32_t buf0 = dfb_row.get_write_ptr();
    const uint32_t stride = (row_bytes + l1_align - 1) & ~(l1_align - 1);
    for (uint32_t i = 0; i < row_count; ++i) {
        const uint32_t dst = idx[i];
        if (dst == 0xFFFFFFFFu) {
            continue;
        }
        const uint32_t l1 = buf0 + (i & 1) * stride;
        noc.async_read(s_src, CoreLocalMem<uint32_t>(l1), row_bytes, {.page_id = row_start + i, .offset_bytes = 0}, {});
        noc.async_read_barrier();
        // the previous write from this half of the buffer finished at the barrier two rows ago
        noc.async_write(CoreLocalMem<uint32_t>(l1), s_cache, row_bytes, {}, {.page_id = dst, .offset_bytes = 0});
        noc.async_writes_flushed();
    }
    noc.async_write_barrier();
}
