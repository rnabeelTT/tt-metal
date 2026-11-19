// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fused hyper-connection (decode), reader (RISCV_1): core c owns hidden tiles [c J, c J + J) of every stream.
// Reads this core's X tiles (stream-major: s * J + jj, X page s * HT + c J + jj), and (POST) the previous sublayer's
// normed output tiles + the previous HC's weight tile, and (PRE) this core's fn tiles (FP32, same page index as X), the
// matmul constants (ones + 8 column selectors) and the FP32 value constants (scale row, base row, affine row).
//
// runtime args: [x, o, wp, fn, mm, cv, c, fn_lo]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t o_addr = get_arg_val<uint32_t>(1);
    const uint32_t wp_addr = get_arg_val<uint32_t>(2);
    const uint32_t fn_addr = get_arg_val<uint32_t>(3);
    const uint32_t mm_addr = get_arg_val<uint32_t>(4);
    const uint32_t cv_addr = get_arg_val<uint32_t>(5);
    const uint32_t c = get_arg_val<uint32_t>(6);

    constexpr uint32_t J = get_compile_time_arg_val(0);
    constexpr uint32_t HT = get_compile_time_arg_val(1);
    constexpr uint32_t POST = get_compile_time_arg_val(2);
    constexpr uint32_t PRE = get_compile_time_arg_val(3);
    constexpr uint32_t cb_x = get_compile_time_arg_val(4);
    constexpr uint32_t cb_o = get_compile_time_arg_val(5);
    constexpr uint32_t cb_wp = get_compile_time_arg_val(6);
    constexpr uint32_t cb_fn = get_compile_time_arg_val(7);
    constexpr uint32_t cb_mm = get_compile_time_arg_val(8);
    constexpr uint32_t cb_cv = get_compile_time_arg_val(9);
    constexpr uint32_t NMM = get_compile_time_arg_val(10);
    constexpr uint32_t NCV = get_compile_time_arg_val(11);
    constexpr uint32_t cb_wpc = get_compile_time_arg_val(12);
    constexpr uint32_t cb_fnlo = get_compile_time_arg_val(13);
    constexpr uint32_t FNLO = get_compile_time_arg_val(14);
    constexpr auto x_args = TensorAccessorArgs<15>();
    constexpr auto o_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto wp_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto fn_args = TensorAccessorArgs<wp_args.next_compile_time_args_offset()>();
    constexpr auto mm_args = TensorAccessorArgs<fn_args.next_compile_time_args_offset()>();
    constexpr auto cv_args = TensorAccessorArgs<mm_args.next_compile_time_args_offset()>();
    constexpr auto fl_args = TensorAccessorArgs<cv_args.next_compile_time_args_offset()>();
    constexpr uint32_t BF = 2048, F32 = 4096;

    const auto s_x = TensorAccessor(x_args, x_addr, BF);
    cb_reserve_back(cb_x, 4 * J);
    uint32_t l1 = get_write_ptr(cb_x);
    for (uint32_t s = 0; s < 4; ++s) {
        for (uint32_t jj = 0; jj < J; ++jj) {
            noc_async_read(s_x.get_noc_addr(s * HT + c * J + jj), l1, BF);
            l1 += BF;
        }
    }
    if constexpr (POST) {
        const auto s_o = TensorAccessor(o_args, o_addr, BF);
        cb_reserve_back(cb_o, J);
        l1 = get_write_ptr(cb_o);
        for (uint32_t jj = 0; jj < J; ++jj) {
            noc_async_read(s_o.get_noc_addr(c * J + jj), l1, BF);
            l1 += BF;
        }
        const auto s_wp = TensorAccessor(wp_args, wp_addr, F32);
        cb_reserve_back(cb_wp, 1);
        noc_async_read(s_wp.get_noc_addr(0), get_write_ptr(cb_wp), F32);
        cb_reserve_back(cb_wpc, 1);  // the same tile for FP32 copies (an unpack-to-dest CB)
        noc_async_read(s_wp.get_noc_addr(0), get_write_ptr(cb_wpc), F32);
    }
    // the matmul constants (ones, selectors) serve POST (the post-weight columns) and PRE
    const auto s_mm = TensorAccessor(mm_args, mm_addr, F32);
    cb_reserve_back(cb_mm, NMM);
    l1 = get_write_ptr(cb_mm);
    for (uint32_t t = 0; t < NMM; ++t) {
        noc_async_read(s_mm.get_noc_addr(t), l1, F32);
        l1 += F32;
    }
    if constexpr (PRE) {
        const auto s_fn = TensorAccessor(fn_args, fn_addr, F32);
        cb_reserve_back(cb_fn, 4 * J);
        l1 = get_write_ptr(cb_fn);
        for (uint32_t s = 0; s < 4; ++s) {
            for (uint32_t jj = 0; jj < J; ++jj) {
                noc_async_read(s_fn.get_noc_addr(s * HT + c * J + jj), l1, F32);
                l1 += F32;
            }
        }
        if constexpr (FNLO) {
            const auto s_fl = TensorAccessor(fl_args, get_arg_val<uint32_t>(7), BF);
            cb_reserve_back(cb_fnlo, 4 * J);
            l1 = get_write_ptr(cb_fnlo);
            for (uint32_t s = 0; s < 4; ++s) {
                for (uint32_t jj = 0; jj < J; ++jj) {
                    noc_async_read(s_fl.get_noc_addr(s * HT + c * J + jj), l1, BF);
                    l1 += BF;
                }
            }
        }
        const auto s_cv = TensorAccessor(cv_args, cv_addr, F32);
        cb_reserve_back(cb_cv, NCV);
        l1 = get_write_ptr(cb_cv);
        for (uint32_t t = 0; t < NCV; ++t) {
            noc_async_read(s_cv.get_noc_addr(t), l1, F32);
            l1 += F32;
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, 4 * J);
    if constexpr (POST) {
        cb_push_back(cb_o, J);
        cb_push_back(cb_wp, 1);
        cb_push_back(cb_wpc, 1);
    }
    cb_push_back(cb_mm, NMM);
    if constexpr (PRE) {
        cb_push_back(cb_fn, 4 * J);
        if constexpr (FNLO) {
            cb_push_back(cb_fnlo, 4 * J);
        }
        cb_push_back(cb_cv, NCV);
    }
}
