// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Multi-row paged KV update, writer: per cache tile row (2, see kvrows_reader.cpp), once compute has untilized it,
// every row whose position lands in it gets this core's head row of that row's input tile row (heads as tile rows:
// two 32-byte face-row reads per tile, L1 -> L1), then compute tilizes the block and the Wt tiles go back to the cache.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t cache_addr = get_arg_val<uint32_t>(0);
    const uint32_t in_addr = get_arg_val<uint32_t>(1);
    const uint32_t head = get_arg_val<uint32_t>(2);

    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(2);
    constexpr uint32_t cb_ucache = get_compile_time_arg_val(3);
    constexpr uint32_t cb_ucache2 = get_compile_time_arg_val(4);
    constexpr uint32_t cb_out = get_compile_time_arg_val(5);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(6);   // one untilized row: Wt * 32 values
    constexpr uint32_t elem_bytes = get_compile_time_arg_val(7);  // 2 (bf16 untilized block)
    constexpr auto cache_args = TensorAccessorArgs<8>();
    constexpr auto in_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
    constexpr uint32_t BF_TILE = 2048, BF_FACE = 512, BF_ROW = 32;

    const uint32_t tile_bytes = get_tile_size(cb_out);
    const auto s_cache = TensorAccessor(cache_args, cache_addr, tile_bytes);
    const auto s_in = TensorAccessor(in_args, in_addr, BF_TILE);

    cb_wait_front(cb_meta, 1);
    volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_meta));
    const uint32_t vt0 = meta[4];
    uint32_t need[2] = {meta[0], meta[2]}, ids[2] = {meta[1], meta[3]};
    int32_t pos[n];
    for (uint32_t u = 0; u < n; ++u) {
        pos[u] = (int32_t)meta[5 + u];
    }
    cb_pop_front(cb_meta, 1);

    const uint32_t src_row = (head >> 4) * 2 * BF_FACE + (head & 15) * BF_ROW;  // face 0 / 2 row of the head
    for (uint32_t g = 0; g < 2; ++g) {
        cb_wait_front(cb_ucache, Wt);
        cb_reserve_back(cb_ucache2, Wt);
        const uint32_t blk = get_read_ptr(cb_ucache);
        if (need[g]) {
            for (uint32_t u = 0; u < n; ++u) {
                if (pos[u] < 0 || (uint32_t)pos[u] / 32 != vt0 + g) {
                    continue;
                }
                const uint32_t dst_row = blk + ((uint32_t)pos[u] & 31) * row_bytes;
                for (uint32_t w = 0; w < Wt; ++w) {
                    for (uint32_t c = 0; c < 2; ++c) {
                        noc_async_read(
                            s_in.get_noc_addr(u * Wt + w, src_row + c * BF_FACE),
                            dst_row + (w * 32 + c * 16) * elem_bytes,
                            BF_ROW);
                    }
                }
            }
            noc_async_read_barrier();
        }
        cb_push_back(cb_ucache2, Wt);
        cb_pop_front(cb_ucache, Wt);

        cb_wait_front(cb_out, Wt);
        if (need[g]) {
            uint32_t src = get_read_ptr(cb_out);
            for (uint32_t w = 0; w < Wt; ++w) {
                noc_async_write(src, s_cache.get_noc_addr(ids[g] + w), tile_bytes);
                src += tile_bytes;
            }
            noc_async_write_barrier();
        }
        cb_pop_front(cb_out, Wt);
    }
}
