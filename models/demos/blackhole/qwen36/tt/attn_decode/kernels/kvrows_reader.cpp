// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Multi-row paged KV update, reader (one core per (K or V, head)): the n rows' positions (diagonal of the [n, n]
// index mask, -1 = skip), the page table row of the first written row, then the two cache tile rows the positions can
// span (virtual tile rows vt0 = min(pos) / 32 and vt0 + 1) into cb_cache, Wt tiles each (pushed unread when no row
// lands in one). The plan goes to the writer through cb_meta: [need0, id0, need1, id1, vt0, pos[0..n)].

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t cache_addr = get_arg_val<uint32_t>(0);
    const uint32_t idx_addr = get_arg_val<uint32_t>(1);
    const uint32_t pt_addr = get_arg_val<uint32_t>(2);
    const uint32_t head = get_arg_val<uint32_t>(3);

    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t num_heads = get_compile_time_arg_val(2);
    constexpr uint32_t block_size = get_compile_time_arg_val(3);
    constexpr uint32_t idx_row_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t pt_row_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t cb_cache = get_compile_time_arg_val(6);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(7);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(8);
    constexpr auto cache_args = TensorAccessorArgs<9>();
    constexpr auto idx_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
    constexpr auto pt_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();
    constexpr uint32_t ROW_STRIDE = 64;  // idx rows land 64-byte aligned in the scratch CB
    constexpr uint32_t block_t = block_size / 32;

    const uint32_t tile_bytes = get_tile_size(cb_cache);
    const auto s_cache = TensorAccessor(cache_args, cache_addr, tile_bytes);
    const auto s_idx = TensorAccessor(idx_args, idx_addr, idx_row_bytes);
    const auto s_pt = TensorAccessor(pt_args, pt_addr, pt_row_bytes);

    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    for (uint32_t u = 0; u < n; ++u) {
        noc_async_read(s_idx.get_noc_addr(u), scr + u * ROW_STRIDE, idx_row_bytes);
    }
    noc_async_read_barrier();
    int32_t pos[n];
    int32_t first = -1, lo = 0x7fffffff;
    for (uint32_t u = 0; u < n; ++u) {
        pos[u] = reinterpret_cast<volatile tt_l1_ptr int32_t*>(scr + u * ROW_STRIDE)[u];
        if (pos[u] >= 0) {
            if (first < 0) {
                first = u;
            }
            if (pos[u] < lo) {
                lo = pos[u];
            }
        }
    }
    const uint32_t pt_l1 = scr + n * ROW_STRIDE;
    if (first >= 0) {
        noc_async_read(s_pt.get_noc_addr(first), pt_l1, pt_row_bytes);
        noc_async_read_barrier();
    }
    volatile tt_l1_ptr uint32_t* pt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pt_l1);

    cb_reserve_back(cb_meta, 1);
    volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta));
    const uint32_t vt0 = first >= 0 ? (uint32_t)lo / 32 : 0;
    uint32_t ids[2], needs[2];
    for (uint32_t g = 0; g < 2; ++g) {
        const uint32_t vt = vt0 + g;
        uint32_t need = 0;
        for (uint32_t u = 0; u < n; ++u) {
            if (pos[u] >= 0 && (uint32_t)pos[u] / 32 == vt) {
                need = 1;
            }
        }
        ids[g] = 0;
        if (need) {
            const uint32_t phys = pt[vt / block_t];
            ids[g] = (phys * num_heads + head) * block_t * Wt + (vt % block_t) * Wt;
        }
        needs[g] = need;
        meta[2 * g] = need;
        meta[2 * g + 1] = ids[g];
    }
    meta[4] = vt0;
    for (uint32_t u = 0; u < n; ++u) {
        meta[5 + u] = (uint32_t)pos[u];
    }
    cb_push_back(cb_meta, 1);

    for (uint32_t g = 0; g < 2; ++g) {
        cb_reserve_back(cb_cache, Wt);
        if (needs[g]) {
            uint32_t dst = get_write_ptr(cb_cache);
            for (uint32_t w = 0; w < Wt; ++w) {
                noc_async_read(s_cache.get_noc_addr(ids[g] + w), dst, tile_bytes);
                dst += tile_bytes;
            }
            noc_async_read_barrier();
        }
        cb_push_back(cb_cache, Wt);
    }
}
