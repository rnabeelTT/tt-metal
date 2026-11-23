// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Streamed decode projection, activation in + output out (BRISC; tt/mstream.py, the LM head's writer with the output
// tile size a compile arg): rows 0 .. rows-1 of the kt BF16 activation tiles into cb_x (rows <= 16: faces 0 / 1, else
// whole tiles), then each output column tile (FP32 or BF16) the compute packs to page out_col0 + i (columns past
// n_tiles: bank padding, not written).
//
// Output column i of this core goes to page out_col0 + i * ostride (1: the per-bank copy's contiguous columns; 8: an
// interleaved weight's bank columns, n = bank + 8 c).
//
// H < 32: the activation / output CB tiles are H x 32 (faces H x 16; rows < H live): rows 0 .. rows-1 of each face
// move between those and the tensors' 32 x 32 tiles (the output rows past `rows` are not written).
//
// runtime args: [x_addr, out_addr, out_col0, ncols]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_col0 = get_arg_val<uint32_t>(2);
    const uint32_t ncols = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t kt = get_compile_time_arg_val(2);
    constexpr uint32_t rows = get_compile_time_arg_val(3);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t OB = get_compile_time_arg_val(5);  // an output tile: FP32 4096 or BF16 2048
    constexpr uint32_t ostride = get_compile_time_arg_val(6);  // output page stride between this core's columns
    constexpr uint32_t H = get_compile_time_arg_val(7);  // the CB tiles' rows (32: whole tiles)
    constexpr auto x_args = TensorAccessorArgs<8>();
    constexpr auto o_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    static_assert(rows <= 32, "one tile row");
    constexpr uint32_t row_bytes = rows * 32;

    if (ncols == 0) {
        return;
    }
    cb_reserve_back(cb_x, kt);
    const uint32_t xl1 = get_write_ptr(cb_x);
    const auto s_x = TensorAccessor(x_args, x_addr, 2048);
    constexpr uint32_t XT = H * 64;  // an activation CB tile (bf16)
    for (uint32_t k = 0; k < kt; ++k) {
        const uint64_t src = s_x.get_noc_addr(k);
        if constexpr (rows > 16) {  // (all four faces)
            noc_async_read(src, xl1 + k * XT, 2048);
        } else {
            noc_async_read(src, xl1 + k * XT, row_bytes);
            noc_async_read(src + 512, xl1 + k * XT + XT / 2, row_bytes);
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, kt);

    const auto s_o = TensorAccessor(o_args, out_addr, OB);
    for (uint32_t i = 0; i < ncols; ++i) {
        cb_wait_front(cb_out, 1);
        if (out_col0 + i * ostride < n_tiles) {
            const uint64_t dst = s_o.get_noc_addr(out_col0 + i * ostride);
            if constexpr (H == 32) {
                noc_async_write(get_read_ptr(cb_out), dst, OB);
            } else {  // rows 0 .. rows-1 of faces 0 / 1 (E = OB / 1024 bytes per element)
                constexpr uint32_t E = OB / 1024;
                noc_async_write(get_read_ptr(cb_out), dst, rows * 16 * E);
                noc_async_write(get_read_ptr(cb_out) + H * 16 * E, dst + 256 * E, rows * 16 * E);
            }
            noc_async_writes_flushed();
        }
        cb_pop_front(cb_out, 1);
    }
    noc_async_write_barrier();
}
