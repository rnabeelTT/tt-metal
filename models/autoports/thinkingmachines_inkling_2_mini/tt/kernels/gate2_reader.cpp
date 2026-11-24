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

// rows 0 .. T-1 of a tile (faces 0 / 1) into a TH x 32 CB tile (faces min(TH, 16) x 16)
template <uint32_t T, uint32_t E, uint32_t TH>
inline void read_rows(uint64_t src, uint32_t dst) {
    constexpr uint32_t FACE = 256 * E;  // (rows 0 .. T-1 of a face are contiguous: one read per face)
    noc_async_read(src, dst, T * 16 * E);
    noc_async_read(src + FACE, dst + (TH < 16 ? TH : 16) * 16 * E, T * 16 * E);
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
    constexpr uint32_t T = G2_ROWS, TH = G2_TH;  // TH: the rows of the CB tiles of token rows (8 / 16 / 32)
    constexpr uint32_t FT = TH * 128, BT = TH * 64;  // (a CB tile of token rows; weights / tensors: 32 x 32)
    constexpr auto x_args = TensorAccessorArgs<16>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto m_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();  // (m: L1 or DRAM)
    auto ca = [](uint32_t i) { return get_common_arg_val<uint32_t>(i); };
#ifndef G2_X0
#define G2_X0 0
#define G2_Y0 0
#endif
    // this core's slot in its rectangle (origin G2_X0, G2_Y0; GateKernel2.multi runs several rectangles)
    const uint32_t c = ((uint32_t)get_absolute_logical_y() - G2_Y0) * G2_CX + (uint32_t)get_absolute_logical_x() - G2_X0;
    const uint32_t t0 = c * PER;
    const auto ah = TensorAccessor(x_args, ca(0), F32);
    const auto am = TensorAccessor(m_args, ca(1), F32);
    const auto awd = TensorAccessor(w_args, ca(2), BF);
    const auto awfg = TensorAccessor(w_args, ca(3), BF);
    const auto aones = TensorAccessor(w_args, ca(4), BF);
    const auto asel = TensorAccessor(w_args, ca(5), BF);

    DeviceZoneScopedN("G2_R_ALL");
    cb_reserve_back(cb_h, PER);
    cb_reserve_back(cb_m, PER);
    for (uint32_t t = 0; t < PER; ++t) {
        read_rows<T, 4, TH>(ah.get_noc_addr(t0 + t), get_write_ptr(cb_h) + t * FT);
        read_rows<T, 4, TH>(am.get_noc_addr(t0 + t), get_write_ptr(cb_m) + t * FT);
    }
    cb_reserve_back(cb_ones, 1);
    if constexpr (LOCAL_ONES) {  // (the bf16 ones tile written here: ~90 cores reading one DRAM page is a hotspot)
        volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ones));
        for (uint32_t w = 0; w < 512; ++w) {
            o[w] = 0x3F803F80u;
        }
    } else {
        noc_async_read(aones.get_noc_addr(0), get_write_ptr(cb_ones), BF);
    }
    noc_async_read_barrier();
    cb_push_back(cb_h, PER);
    cb_push_back(cb_m, PER);
    cb_push_back(cb_ones, 1);
    cb_reserve_back(cb_sel, NSEL);
    for (uint32_t k = 0; k < ((LOCAL_ONES && c >= (G2_MR ? 2 * T : T)) ? 0u : NSEL); ++k) {  // (roots only with LOCAL_ONES; row 0 of the 0 / 1 tiles is the only nonzero row)
        if constexpr (TH == 32) {
            noc_async_read(asel.get_noc_addr(k), get_write_ptr(cb_sel) + k * BT, BT);
        } else {
            read_rows<TH, 2, TH>(asel.get_noc_addr(k), get_write_ptr(cb_sel) + k * BT);
        }
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
            read_rows<T, 4, TH>(anw.get_noc_addr(t0 + t), get_write_ptr(cb_nw) + t * FT);
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
