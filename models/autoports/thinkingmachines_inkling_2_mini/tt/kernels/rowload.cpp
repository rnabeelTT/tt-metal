// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Row loads into fp32 TILE tensors (tt/rowload.py): each job copies row j of a ROW_MAJOR fp32 DRAM tensor (one page per
// row, C fp32 values) into row b of an fp32 TILE DRAM tensor [.., B, C] (interleaved, 4096-byte tiles). Row r = b % 32
// of a tile lives in faces (r / 16) * 2 + {0, 1}, 64 bytes each at face row r % 16: two 64-byte writes per tile.
// A prefill's end (BatchedDecoder.load_slot): every application's conv states in one launch, bitwise copies.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr auto args = TensorAccessorArgs<1>();
    constexpr uint32_t TILE_BYTES = 4096;  // fp32 32 x 32
    constexpr uint32_t FACE_BYTES = 1024;  // fp32 16 x 16
    const uint32_t n_jobs = get_arg_val<uint32_t>(0);
    const uint32_t l1 = get_write_ptr(cb);
    uint32_t ai = 1;
    for (uint32_t j = 0; j < n_jobs; ++j) {
        const uint32_t src = get_arg_val<uint32_t>(ai);
        const uint32_t src_row = get_arg_val<uint32_t>(ai + 1);
        const uint32_t src_page = get_arg_val<uint32_t>(ai + 2);  // bytes per source row (padded)
        const uint32_t dst = get_arg_val<uint32_t>(ai + 3);
        const uint32_t b = get_arg_val<uint32_t>(ai + 4);
        const uint32_t ct = get_arg_val<uint32_t>(ai + 5);  // tile columns (C / 32)
        ai += 6;
        const auto as = TensorAccessor(args, src, src_page);
        const auto ad = TensorAccessor(args, dst, TILE_BYTES);
        noc_async_read(as.get_noc_addr(src_row), l1, ct * 128);
        noc_async_read_barrier();
        const uint32_t r = b & 31;
        const uint32_t t0 = (b >> 5) * ct;
        const uint32_t off = (r >> 4) * 2 * FACE_BYTES + (r & 15) * 64;
        for (uint32_t c = 0; c < ct; ++c) {
            const uint32_t s = l1 + c * 128;
            noc_async_write(s, ad.get_noc_addr(t0 + c, off), 64);
            noc_async_write(s + 64, ad.get_noc_addr(t0 + c, off + FACE_BYTES), 64);
        }
        noc_async_write_barrier();
    }
}
