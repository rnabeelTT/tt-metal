// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Prefill row norm (tt/rown.py), reader: the reduce scaler (bf16, 1.0 in row 0 of each face) and the eps column tile
// (bf16, eps in column 0) once, the weight's Wt bf16 tiles once (ROWN_GAMMA), then per row tile all Wt input tiles
// (pushed per 4-tile block; the compute keeps the whole row).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // common args (per call): x, weight, eps bits, Qrep (ROWN_LG, [H, NQ * 32] fp32); per core: its row tiles
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t g_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t eps = get_common_arg_val<uint32_t>(2);
    const uint32_t q_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t r0 = get_arg_val<uint32_t>(0);
    const uint32_t nr = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(1);
    constexpr uint32_t cb_eps = get_compile_time_arg_val(2);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(3);
    constexpr uint32_t Wt = get_compile_time_arg_val(4);
    constexpr uint32_t xb = get_compile_time_arg_val(5);  // input tile bytes
    constexpr uint32_t BLK = 4;
    constexpr uint32_t cb_q = get_compile_time_arg_val(6);
    constexpr uint32_t NQ = get_compile_time_arg_val(7);
    constexpr uint32_t cb_fus = get_compile_time_arg_val(8);
    constexpr auto x_args = TensorAccessorArgs<9>();
    const auto ax = TensorAccessor(x_args, x_addr, xb);

    // scaler: zeros, 1.0 (bf16 0x3f80) in row 0 of each of the 4 faces
    cb_reserve_back(cb_scaler, 1);
    {
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_scaler));
        for (uint32_t i = 0; i < 512; ++i) {
            p[i] = 0;
        }
        for (uint32_t f = 0; f < 4; ++f) {
            for (uint32_t j = 0; j < 8; ++j) {
                p[f * 128 + j] = 0x3f803f80u;
            }
        }
    }
    cb_push_back(cb_scaler, 1);
    // eps: bf16 (the fp32 bits' upper half) in column 0 of faces 0 / 2 (generate_bcast_col_scalar)
    cb_reserve_back(cb_eps, 1);
    {
        volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_eps));
        for (uint32_t i = 0; i < 1024; ++i) {
            p[i] = 0;
        }
        for (uint32_t k = 0; k < 4; k += 2) {
            for (uint32_t j = 0; j < 256; j += 16) {
                p[(k << 8) + j] = static_cast<uint16_t>(eps >> 16);
            }
        }
    }
    cb_push_back(cb_eps, 1);
    constexpr auto g_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
#ifdef ROWN_LG
    constexpr auto q_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    const auto aq = TensorAccessor(q_args, q_addr, 4096);
#endif
#ifdef ROWN_GAMMA
    auto load_gamma = [&]() {
        // the weight as one row-major bf16 row [H] (the checkpoint's values), staged in cb_fus (unused until the output
        // pass), into row 0 of faces 0 / 1 of the bf16 gamma tiles (all mul_tiles_bcast_rows reads)
        const auto ag = TensorAccessor(g_args, g_addr, Wt * 64);
        cb_reserve_back(cb_gamma, Wt);
        const uint32_t p = get_write_ptr(cb_gamma);
        const uint32_t stage = get_write_ptr(cb_fus);
        noc_async_read(ag.get_noc_addr(0), stage, Wt * 64);
        noc_async_read_barrier();
        const uint32_t* src = reinterpret_cast<const uint32_t*>(stage);
        for (uint32_t t = 0; t < Wt; ++t) {
            uint32_t* dst = reinterpret_cast<uint32_t*>(p + t * 2048);
            for (uint32_t j = 0; j < 8; ++j) {
                dst[j] = src[t * 16 + j];
                dst[128 + j] = src[t * 16 + 8 + j];
            }
        }
        cb_push_back(cb_gamma, Wt);
    };
#endif
    for (uint32_t r = r0; r < r0 + nr; ++r) {
        for (uint32_t b = 0; b < Wt; b += BLK) {
            cb_reserve_back(cb_in, BLK);
            const uint32_t p = get_write_ptr(cb_in);
            for (uint32_t j = 0; j < BLK; ++j) {
                noc_async_read(ax.get_noc_addr(r * Wt + b + j), p + j * xb, xb);
            }
            noc_async_read_barrier();
            cb_push_back(cb_in, BLK);
        }
#ifdef ROWN_GAMMA
        if (r == r0) {  // (after the first row: the compute needs the weight only for its output pass)
            load_gamma();
        }
#endif
#ifdef ROWN_LG
        // the query tiles of each block (K tiles b .. b + 3, NQ columns), in the compute's output order
        for (uint32_t b = 0; b < Wt; b += BLK) {
            cb_reserve_back(cb_q, BLK * NQ);
            const uint32_t p = get_write_ptr(cb_q);
            for (uint32_t j = 0; j < BLK * NQ; ++j) {
                noc_async_read(aq.get_noc_addr(b * NQ + j), p + j * 4096, 4096);
            }
            noc_async_read_barrier();
            cb_push_back(cb_q, BLK * NQ);
        }
#endif
    }
}
