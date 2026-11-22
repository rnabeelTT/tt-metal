// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Message gate, round-reduced version (tt/gate.py GateKernel2), reader: this core's PER hidden tiles of h and m (live
// rows, fp32), its rows of Wd ([2H, R*32] bf16, tiles [half][t][n]), its columns of Wfg ([R*32, 2H] bf16: forget
// [k][t], gain [k][t]), the ones tile and the selection tiles; NORM: its norm weight rows ([1, 1, 32, H] fp32); ROUTER:
// its rows of the router weight's bf16 hi / lo halves ([H, E] each; tiles [half][t][e]).
//
// common args: h, m, wd, wfg, ones, sel, nw, router hi, router lo

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

template <uint32_t T, uint32_t E>
inline void read_rows(uint64_t src, uint32_t dst) {
    constexpr uint32_t HALF = 16 * E, FACE = 256 * E;
    for (uint32_t r = 0; r < T; ++r) {
        noc_async_read(src + r * HALF, dst + r * HALF, HALF);
        noc_async_read(src + FACE + r * HALF, dst + FACE + r * HALF, HALF);
    }
}

void kernel_main() {
    constexpr uint32_t cb_h = get_compile_time_arg_val(0);
    constexpr uint32_t cb_m = get_compile_time_arg_val(1);
    constexpr uint32_t cb_wd = get_compile_time_arg_val(2);
    constexpr uint32_t cb_wf = get_compile_time_arg_val(3);
    constexpr uint32_t cb_wg = get_compile_time_arg_val(4);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(5);
    constexpr uint32_t cb_sel = get_compile_time_arg_val(6);
    constexpr uint32_t PER = get_compile_time_arg_val(7);
    constexpr uint32_t R = get_compile_time_arg_val(8);
    constexpr uint32_t HT = get_compile_time_arg_val(9);
    constexpr uint32_t NSEL = get_compile_time_arg_val(10);
    constexpr uint32_t cb_nw = get_compile_time_arg_val(11);
    constexpr uint32_t cb_rw = get_compile_time_arg_val(12);
    constexpr uint32_t NORM = get_compile_time_arg_val(13);
    constexpr uint32_t ROUTER = get_compile_time_arg_val(14);
    constexpr uint32_t ET = get_compile_time_arg_val(15);
    constexpr uint32_t F32 = 4096, BF = 2048;
    constexpr uint32_t T = G2_ROWS;
    constexpr auto x_args = TensorAccessorArgs<16>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    auto ca = [](uint32_t i) { return get_common_arg_val<uint32_t>(i); };
    const uint32_t c = (uint32_t)get_absolute_logical_y() * G2_CX + (uint32_t)get_absolute_logical_x();
    const uint32_t t0 = c * PER;
    const auto ah = TensorAccessor(x_args, ca(0), F32);
    const auto am = TensorAccessor(x_args, ca(1), F32);
    const auto awd = TensorAccessor(w_args, ca(2), BF);
    const auto awfg = TensorAccessor(w_args, ca(3), BF);
    const auto aones = TensorAccessor(w_args, ca(4), BF);
    const auto asel = TensorAccessor(w_args, ca(5), BF);

    DeviceZoneScopedN("G2_R_ALL");
    cb_reserve_back(cb_h, PER);
    cb_reserve_back(cb_m, PER);
    for (uint32_t t = 0; t < PER; ++t) {
        read_rows<T, 4>(ah.get_noc_addr(t0 + t), get_write_ptr(cb_h) + t * F32);
        read_rows<T, 4>(am.get_noc_addr(t0 + t), get_write_ptr(cb_m) + t * F32);
    }
    cb_reserve_back(cb_ones, 1);
    noc_async_read(aones.get_noc_addr(0), get_write_ptr(cb_ones), BF);
    noc_async_read_barrier();
    cb_push_back(cb_h, PER);
    cb_push_back(cb_m, PER);
    cb_push_back(cb_ones, 1);
    cb_reserve_back(cb_sel, NSEL);
    for (uint32_t k = 0; k < NSEL; ++k) {
        noc_async_read(asel.get_noc_addr(k), get_write_ptr(cb_sel) + k * BF, BF);
    }
    cb_reserve_back(cb_wd, 2 * PER * R);
    const uint32_t w = get_write_ptr(cb_wd);
    for (uint32_t half = 0; half < 2; ++half) {
        for (uint32_t t = 0; t < PER; ++t) {
            for (uint32_t n = 0; n < R; ++n) {
                noc_async_read(awd.get_noc_addr((half * HT + t0 + t) * R + n), w + ((half * PER + t) * R + n) * BF, BF);
            }
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_sel, NSEL);
    cb_push_back(cb_wd, 2 * PER * R);
    cb_reserve_back(cb_wf, R * PER);
    cb_reserve_back(cb_wg, R * PER);
    for (uint32_t k = 0; k < R; ++k) {
        for (uint32_t t = 0; t < PER; ++t) {
            noc_async_read(awfg.get_noc_addr(k * 2 * HT + t0 + t), get_write_ptr(cb_wf) + (k * PER + t) * BF, BF);
            noc_async_read(awfg.get_noc_addr(k * 2 * HT + HT + t0 + t), get_write_ptr(cb_wg) + (k * PER + t) * BF, BF);
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_wf, R * PER);
    cb_push_back(cb_wg, R * PER);
    if constexpr (NORM) {
        const auto anw = TensorAccessor(w_args, ca(6), F32);
        cb_reserve_back(cb_nw, PER);
        for (uint32_t t = 0; t < PER; ++t) {
            read_rows<T, 4>(anw.get_noc_addr(t0 + t), get_write_ptr(cb_nw) + t * F32);
        }
        noc_async_read_barrier();
        cb_push_back(cb_nw, PER);
    }
    if constexpr (ROUTER) {
        cb_reserve_back(cb_rw, 2 * PER * ET);
        const uint32_t rw = get_write_ptr(cb_rw);
        for (uint32_t half = 0; half < 2; ++half) {
            const auto ar = TensorAccessor(w_args, ca(7 + half), BF);
            for (uint32_t t = 0; t < PER; ++t) {
                for (uint32_t e = 0; e < ET; ++e) {
                    noc_async_read(ar.get_noc_addr((t0 + t) * ET + e), rw + ((half * PER + t) * ET + e) * BF, BF);
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_rw, 2 * PER * ET);
    }
}
