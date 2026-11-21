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
    static_assert(rows <= 16 || rows == 32, "faces 0 / 1, or whole tiles");
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
#ifdef X_SPLIT
    // the activation as two K parts (ops.py x=(xa, xb)): tiles [0, X_KT1) of xa, then xb's
    constexpr auto xb_args = TensorAccessorArgs<XB_CTA_OFFSET>();
    const auto s_xb = TensorAccessor(xb_args, get_arg_val<uint32_t>(X_B_ARG), 2048);
#endif
#ifdef X_HEADS
    // per-head projection (ops.py heads=): this core's columns are head h's; its activation tiles [h X_XS, + kt)
    const uint32_t x0 = (out_col0 / X_NH) * X_XS;
#endif
    for (uint32_t k = 0; k < kt; ++k) {
#ifdef X_SPLIT
        const uint64_t src = k < X_KT1 ? s_x.get_noc_addr(k) : s_xb.get_noc_addr(k - X_KT1);
#elif defined(X_HEADS)
        const uint64_t src = s_x.get_noc_addr(x0 + k);
#else
        const uint64_t src = s_x.get_noc_addr(k);
#endif
        if constexpr (rows == 32) {  // whole tiles: every output row defined (ports that carry 32-row padded passes)
            noc_async_read(src, xl1 + k * 2048, 2048);
        } else {
            noc_async_read(src, xl1 + k * 2048, row_bytes);
            noc_async_read(src + 512, xl1 + k * 2048 + 512, row_bytes);
        }
    }
#endif
    noc_async_read_barrier();
    cb_push_back(cb_x, kt);

#ifndef OUT_PAGE_BYTES
#define OUT_PAGE_BYTES 2048
#endif
    const auto s_o = TensorAccessor(o_args, out_addr, OUT_PAGE_BYTES);
#ifdef OUT_SPLIT
    // the output as up to 5 column parts (ops.py splits=): part k starts at column tile OBk (unused parts: 1 << 20)
    constexpr auto o1_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto o2_args = TensorAccessorArgs<o1_args.next_compile_time_args_offset()>();
    constexpr auto o3_args = TensorAccessorArgs<o2_args.next_compile_time_args_offset()>();
    constexpr auto o4_args = TensorAccessorArgs<o3_args.next_compile_time_args_offset()>();
    const auto s_o1 = TensorAccessor(o1_args, get_arg_val<uint32_t>(4), OUT_PAGE_BYTES);
    const auto s_o2 = TensorAccessor(o2_args, get_arg_val<uint32_t>(5), OUT_PAGE_BYTES);
    const auto s_o3 = TensorAccessor(o3_args, get_arg_val<uint32_t>(6), OUT_PAGE_BYTES);
    const auto s_o4 = TensorAccessor(o4_args, get_arg_val<uint32_t>(7), OUT_PAGE_BYTES);
    auto dst_addr = [&](uint32_t p) -> uint64_t {
        if (p >= OB4) {
            return s_o4.get_noc_addr(p - OB4);
        }
        if (p >= OB3) {
            return s_o3.get_noc_addr(p - OB3);
        }
        if (p >= OB2) {
            return s_o2.get_noc_addr(p - OB2);
        }
        if (p >= OB1) {
            return s_o1.get_noc_addr(p - OB1);
        }
        return s_o.get_noc_addr(p);
    };
#else
    auto dst_addr = [&](uint32_t p) -> uint64_t { return s_o.get_noc_addr(p); };
#endif
#ifdef ZERO_PAD
    // only rows 0 .. rows-1 of the activation were read (the rest of cb_x is stale L1): zero output rows >= rows before
    // the write, so padded rows stay defined (consumers that mix rows: one-hot staging matmuls, X X^T)
    static_assert(rows <= 16, "ZERO_PAD: faces 0 / 1 rows");
    constexpr uint32_t FACE = OUT_PAGE_BYTES / 4, ROWB = FACE / 16;
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    auto zero = [&](uint32_t dst, uint32_t n) {
        while (n) {
            const uint32_t k = n < MEM_ZEROS_SIZE ? n : MEM_ZEROS_SIZE;
            noc_async_read(zeros, dst, k);
            dst += k;
            n -= k;
        }
    };
#endif
    for (uint32_t i = 0; i < ncols; ++i) {
        cb_wait_front(cb_out, 1);
#ifdef ZERO_PAD
        {
            const uint32_t t = get_read_ptr(cb_out);
            zero(t + rows * ROWB, (16 - rows) * ROWB);         // face 0, rows rows..15
            zero(t + FACE + rows * ROWB, (16 - rows) * ROWB);  // face 1
            zero(t + 2 * FACE, 2 * FACE);                      // faces 2, 3 (rows 16..31)
            noc_async_read_barrier();
        }
#endif
        if (out_col0 + i < n_tiles) {
            noc_async_write(get_read_ptr(cb_out), dst_addr(out_col0 + i), OUT_PAGE_BYTES);
            noc_async_writes_flushed();
        }
        cb_pop_front(cb_out, 1);
    }
    noc_async_write_barrier();
}
