// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode linear, activation in + output out (BRISC, NOC1) (decode_linear/ops.py DecodeLinear): reads rows 0 .. B-1 of
// the kt activation tiles (BF16 [32, K] TILE, any TensorAccessor layout; faces 0 / 1 rows only, B <= 16: the matmul's
// other rows are don't-care, their outputs land in output rows nobody reads) into cb_x, then writes each output column
// tile the compute packs to output page `out_col0 + i` (columns past n_tiles are bank padding: not written).
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
    constexpr auto x_args = TensorAccessorArgs<5>();
    constexpr auto o_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    static_assert(rows <= 16, "faces 0 / 1");
    constexpr uint32_t row_bytes = rows * 32;

    if (ncols == 0) {
        return;
    }
    cb_reserve_back(cb_x, kt);
    const uint32_t xl1 = get_write_ptr(cb_x);
#ifdef X_FROM_FLAT
    // fused after a fabric boundary (fabric_boundary/ops.py consumer=): once program semaphore X_SEM is incremented, the
    // normed row is the flat BF16 vector x_addr on the boundary core (NOC X_NOC_X, X_NOC_Y): 32-value group k is row 0
    // of activation tile k
    // One read of the whole flat row per core, then local copies into the tiles: 40 cores x 128 small reads of the
    // boundary core's L1 serialized on its NOC endpoint (the activation arrived 25-40 us after the boundary).
    static_assert(rows == 1, "one row");
    {
        DeviceZoneScopedN("DL_XSEM");
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(X_SEM)), 1);
    }
    cb_reserve_back(CB_XF, 1);
    const uint32_t xf = get_write_ptr(CB_XF);
    noc_async_read(get_noc_addr(X_NOC_X, X_NOC_Y, x_addr), xf, kt * 64);
    noc_async_read_barrier();
    for (uint32_t k = 0; k < kt; ++k) {
        const uint64_t src = get_noc_addr(xf + 64 * k);
        noc_async_read(src, xl1 + k * 2048, 32);
        noc_async_read(src + 32, xl1 + k * 2048 + 512, 32);
    }
#else
    const auto s_x = TensorAccessor(x_args, x_addr, 2048);
    for (uint32_t k = 0; k < kt; ++k) {
        const uint64_t src = s_x.get_noc_addr(k);
        noc_async_read(src, xl1 + k * 2048, row_bytes);
        noc_async_read(src + 512, xl1 + k * 2048 + 512, row_bytes);
    }
#endif
    noc_async_read_barrier();
    cb_push_back(cb_x, kt);

    const auto s_o = TensorAccessor(o_args, out_addr, 2048);
    for (uint32_t i = 0; i < ncols; ++i) {
        cb_wait_front(cb_out, 1);
        if (out_col0 + i < n_tiles) {
            noc_async_write(get_read_ptr(cb_out), s_o.get_noc_addr(out_col0 + i), 2048);
            noc_async_writes_flushed();
        }
        cb_pop_front(cb_out, 1);
    }
    noc_async_write_barrier();
}
