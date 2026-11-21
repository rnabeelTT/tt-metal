// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fused hyper-connection, reader, HALF-TILE decode variant (hc_compute_h.cpp): per-row tiles are read as their rows
// 0..15 (the first half of each tile page: faces 0 / 1); fn / fn_lo / cb_mm full; cb_one = the ones constant's half;
// without POST, X also as full tiles into cb_xf (the in1 of X X^T).
// (Full-tile notes follow.) Fused hyper-connection, reader (RISCV_1): core c of its group owns hidden tiles [c J, c J + J) of every stream.
// Reads (once) the matmul constants (ones + 8 column selectors), (PRE) this core's fn tiles (FP32, rows s * H + its
// hidden columns) and the FP32 value constants (scale row, base row, affine row); then per row tile r of the group
// (r = g, g + G, ... < TR: one 32-row tile row per iteration; decode TR = G = 1) this core's X tiles (stream-major
// s * J + jj, X page (s TR + r) HT + c J + jj) and (POST) the previous sublayer's normed output tiles (page r HT + c J
// + jj) + the previous HC's weight tile (page r).
//
// runtime args: [x, o, wp, fn, mm, cv, c, fn_lo, g]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

#ifndef QF_IN
#define QF_IN 0
#endif
#ifndef COMB
#define COMB 0
#endif
#ifndef EARLY
#define EARLY 0  // decode: the row tile's inputs (X, o, the post selections) first, the constants after
#endif
#ifndef HR
#define HR 16  // rows of the partial tiles (16: faces 0 / 1 whole; 8: their rows 0..7)
#endif

// rows 0..HR-1 of a full tile at NOC address src (element bytes EB) -> a partial HR x 32 tile at dst
template <uint32_t EB>
inline void read_rows(uint64_t src, uint32_t dst) {
    if constexpr (HR == 16) {
        noc_async_read(src, dst, 2 * 256 * EB);
    } else {
        noc_async_read(src, dst, HR * 16 * EB);
        noc_async_read(src + 256 * EB, dst + HR * 16 * EB, HR * 16 * EB);
    }
}

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
    constexpr uint32_t TR = get_compile_time_arg_val(15);  // row tiles of the tensors
    constexpr uint32_t G = get_compile_time_arg_val(16);   // core groups (row-tile stride)
    constexpr auto x_args = TensorAccessorArgs<17>();
    constexpr auto o_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto wp_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto fn_args = TensorAccessorArgs<wp_args.next_compile_time_args_offset()>();
    constexpr auto mm_args = TensorAccessorArgs<fn_args.next_compile_time_args_offset()>();
    constexpr auto cv_args = TensorAccessorArgs<mm_args.next_compile_time_args_offset()>();
    constexpr auto fl_args = TensorAccessorArgs<cv_args.next_compile_time_args_offset()>();
    constexpr uint32_t BF = 2048, F32 = 4096, BFH = HR * 64, F32H = HR * 128;
    constexpr uint32_t cb_xf = 25, cb_one = 26;

    const uint32_t g = get_arg_val<uint32_t>(8);

    // constants (once)
    auto read_consts = [&]() {
#if EARLY
    if constexpr (PRE) {  // fn first (the partial mix needs it first), pushed on its own
        const auto s_fn = TensorAccessor(fn_args, fn_addr, F32);
        cb_reserve_back(cb_fn, 4 * J);
        uint32_t l1 = get_write_ptr(cb_fn);
        for (uint32_t s = 0; s < 4; ++s) {
            for (uint32_t jj = 0; jj < J; ++jj) {
                noc_async_read(s_fn.get_noc_addr(s * HT + c * J + jj), l1, F32);
                l1 += F32;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_fn, 4 * J);
    }
#endif
    const auto s_mm = TensorAccessor(mm_args, mm_addr, F32);
    cb_reserve_back(cb_mm, NMM);
    uint32_t l1 = get_write_ptr(cb_mm);
    for (uint32_t t = 0; t < NMM; ++t) {
        noc_async_read(s_mm.get_noc_addr(t), l1, F32);
        l1 += F32;
    }
    cb_reserve_back(cb_one, 1);
    read_rows<4>(s_mm.get_noc_addr(0), get_write_ptr(cb_one));
    if constexpr (PRE) {
#if !EARLY
        const auto s_fn = TensorAccessor(fn_args, fn_addr, F32);
        cb_reserve_back(cb_fn, 4 * J);
        l1 = get_write_ptr(cb_fn);
        for (uint32_t s = 0; s < 4; ++s) {
            for (uint32_t jj = 0; jj < J; ++jj) {
                noc_async_read(s_fn.get_noc_addr(s * HT + c * J + jj), l1, F32);
                l1 += F32;
            }
        }
#endif
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
            read_rows<4>(s_cv.get_noc_addr(t), l1);
            l1 += F32H;
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_mm, NMM);
    cb_push_back(cb_one, 1);
    if constexpr (PRE) {
#if !EARLY
        cb_push_back(cb_fn, 4 * J);
#endif
        if constexpr (FNLO) {
            cb_push_back(cb_fnlo, 4 * J);
        }
        cb_push_back(cb_cv, NCV);
    }
    };
#if !EARLY
    read_consts();
#endif
    uint32_t l1;

    // per row tile
    const auto s_x = TensorAccessor(x_args, x_addr, BF);
    const auto s_o = TensorAccessor(o_args, o_addr, BF);
    const auto s_wp = TensorAccessor(wp_args, wp_addr, F32);
    for (uint32_t r = g; r < TR; r += G) {
        cb_reserve_back(cb_x, 4 * J);
        l1 = get_write_ptr(cb_x);
        for (uint32_t s = 0; s < 4; ++s) {
            for (uint32_t jj = 0; jj < J; ++jj) {
                read_rows<2>(s_x.get_noc_addr((s * TR + r) * HT + c * J + jj), l1);
                l1 += BFH;
            }
        }
        if constexpr (PRE && !POST && !COMB) {  // X as full tiles: the in1 of X X^T (COMB: the writer shifts them)
            cb_reserve_back(cb_xf, 4 * J);
            l1 = get_write_ptr(cb_xf);
            for (uint32_t s = 0; s < 4; ++s) {
                for (uint32_t jj = 0; jj < J; ++jj) {
                    noc_async_read(s_x.get_noc_addr((s * TR + r) * HT + c * J + jj), l1, BF);
                    l1 += BF;
                }
            }
        }
        if constexpr (POST) {
            cb_reserve_back(cb_o, J);
            l1 = get_write_ptr(cb_o);
            for (uint32_t jj = 0; jj < J; ++jj) {
                read_rows<2>(s_o.get_noc_addr(r * HT + c * J + jj), l1);
                l1 += BFH;
            }
#if QF_IN
            // the post selections Q_s hi (pages 1..4) / lo (5..8) of the previous PRE's weight tensor (one row tile)
            constexpr uint32_t cb_q = 6, cb_ql = 23;
            cb_reserve_back(cb_q, 4);
            cb_reserve_back(cb_ql, 4);
            for (uint32_t i = 0; i < 4; ++i) {
                read_rows<4>(s_wp.get_noc_addr(1 + i), get_write_ptr(cb_q) + i * F32H);
                read_rows<4>(s_wp.get_noc_addr(5 + i), get_write_ptr(cb_ql) + i * F32H);
            }
#else
            cb_reserve_back(cb_wp, 1);
            read_rows<4>(s_wp.get_noc_addr(r), get_write_ptr(cb_wp));
            cb_reserve_back(cb_wpc, 1);  // the same tile for FP32 copies (an unpack-to-dest CB)
            read_rows<4>(s_wp.get_noc_addr(r), get_write_ptr(cb_wpc));
#endif
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, 4 * J);
        if constexpr (PRE && !POST && !COMB) {
            cb_push_back(cb_xf, 4 * J);
        }
        if constexpr (POST) {
            cb_push_back(cb_o, J);
#if QF_IN
            cb_push_back(6, 4);
            cb_push_back(23, 4);
#else
            cb_push_back(cb_wp, 1);
            cb_push_back(cb_wpc, 1);
#endif
        }
    }
#if EARLY
    read_consts();
#endif
}
