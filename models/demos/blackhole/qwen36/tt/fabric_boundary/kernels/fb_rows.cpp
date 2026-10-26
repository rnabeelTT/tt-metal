// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode layer boundary, pack + to rows (fabric_boundary/ops.py: FabricBoundary).
// FB_PACK_ROWS: first packs row 0 of output tiles tile0 .. tile0 + tiles - 1 of this device's partial sum (any TensorAccessor
// layout, BF16) flat into this device's slot of the site's receive buffer on the boundary core (2 x 32 B per tile) and
// increments program semaphore 1 there (the sender publishes the slot once every to-rows core did).
// Then, once the boundary core has the normed
// output (program semaphore 0 incremented by fb_reader.cpp), builds `tiles` 32x32 BF16 tiles of the [1, 1, 32, H]
// output, row 0 = values 32 t .. 32 t + 31 of the flat vector on the boundary core, rows 1..31 zero, and writes them
// to output pages tile0 .. tile0 + tiles - 1 (any TensorAccessor layout). After gpt-oss's boundary_to_rows.cpp.
//
// runtime args: [x_flat_addr, out_addr, tile0, (FB_PACK_ROWS) partial_addr, slot_addr]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t tile0 = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_scr = get_compile_time_arg_val(0);
    constexpr uint32_t tiles = get_compile_time_arg_val(1);
    constexpr uint32_t x_noc_x = get_compile_time_arg_val(2);
    constexpr uint32_t x_noc_y = get_compile_time_arg_val(3);
    constexpr auto out_args = TensorAccessorArgs<4>();
#ifdef FB_PACK_ROWS
    constexpr auto part_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_pk = get_compile_time_arg_val(part_args.next_compile_time_args_offset());
#endif

    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t face_bytes = 512;
    constexpr uint32_t half_row = 32;

#ifdef FB_PACK_ROWS
    {
        // rows 0 .. FB_ROWS-1 of each of my tiles: row r's 64 bytes of tile t at pk + r * 64 * tiles + 64 t, then one write
        // per row into the slot (row r's vector at byte r * slot_row_bytes)
        const auto s_part = TensorAccessor(part_args, get_arg_val<uint32_t>(3), tile_bytes);
        const uint32_t pk = get_write_ptr(cb_pk);
        for (uint32_t t = 0; t < tiles; ++t) {
            const uint64_t src = s_part.get_noc_addr(tile0 + t);
            for (uint32_t r = 0; r < FB_ROWS; ++r) {
                const uint32_t roff = (r / 16) * 1024 + (r % 16) * 32;
                noc_async_read(src + roff, pk + r * 64 * tiles + 64 * t, half_row);
                noc_async_read(src + roff + face_bytes, pk + r * 64 * tiles + 64 * t + half_row, half_row);
            }
        }
        noc_async_read_barrier();
        for (uint32_t r = 0; r < FB_ROWS; ++r) {
            const uint64_t slot = get_noc_addr(x_noc_x, x_noc_y, get_arg_val<uint32_t>(4) + r * FB_ROW_BYTES + 64 * tile0);
            noc_async_write(pk + r * 64 * tiles, slot, 64 * tiles);
        }
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(x_noc_x, x_noc_y, get_semaphore(1)), 1);
    }
#ifdef FB_PACK_ONLY  // a consumer op in the program reads the normed row from the boundary core itself
    noc_async_atomic_barrier();
    return;
#endif
#endif
    // the zero parts of the tiles while the boundary computes
    const uint32_t dst0 = get_write_ptr(cb_scr);
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t t = 0; t < tiles; ++t) {
        const uint32_t base = dst0 + t * tile_bytes;
        for (uint32_t z = 0; z < tile_bytes; z += MEM_ZEROS_SIZE) {
            const uint32_t n = tile_bytes - z < MEM_ZEROS_SIZE ? tile_bytes - z : MEM_ZEROS_SIZE;
            noc_async_read(zeros, base + z, n);
        }
    }
    noc_async_read_barrier();  // the zeros land before the rows (both are NOC reads into the same tiles)
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0)), 1);
#ifdef FB_PACK_ROWS
    // one read per row of this core's 64 * tiles bytes (16 cores x many 32-byte reads of the boundary core's L1
    // serialize on its NOC endpoint), then local copies into the tiles; cb_pk is free again after the pack
    const uint32_t gx = get_write_ptr(cb_pk);
#else
    const uint32_t gx = get_write_ptr(cb_scr) + tiles * tile_bytes;
#endif
    for (uint32_t r = 0; r < FB_ROWS; ++r) {
        noc_async_read(get_noc_addr(x_noc_x, x_noc_y, x_addr + r * FB_ROW_BYTES + tile0 * 2 * half_row),
                       gx + r * 64 * tiles, 64 * tiles);
    }
    noc_async_read_barrier();
    for (uint32_t t = 0; t < tiles; ++t) {
        const uint32_t base = dst0 + t * tile_bytes;
        for (uint32_t r = 0; r < FB_ROWS; ++r) {
            const uint32_t roff = (r / 16) * 1024 + (r % 16) * 32;
            const uint64_t src = get_noc_addr(gx + r * 64 * tiles + 64 * t);
            noc_async_read(src, base + roff, half_row);
            noc_async_read(src + half_row, base + roff + face_bytes, half_row);
        }
    }
    noc_async_read_barrier();
    const auto s_out = TensorAccessor(out_args, out_addr, tile_bytes);
    for (uint32_t t = 0; t < tiles; ++t) {
        noc_async_write(dst0 + t * tile_bytes, s_out.get_noc_addr(tile0 + t), tile_bytes);
    }
    noc_async_write_barrier();
}
