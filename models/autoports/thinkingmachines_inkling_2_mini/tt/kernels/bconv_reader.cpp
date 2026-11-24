// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Batched short conv step (tt/bconv.py), reader: for each of this core's channel tiles, x (one 32-row tile: the B
// slots' rows), the three fp32 state tiles s0 / s1 / s2 and the four fp32 tap tiles w0 / w1 / w2 / wc (every row = the
// tap; wc = w3 + 1, the current tap plus the residual).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#ifdef SEQ_R
#include "seqshift.hpp"
#endif

// BCONV_LIVE (<= 16, tt/bconv.py I2M_BCONV_LIVE): only rows 0 .. LIVE-1 of each tile are read (faces 0 / 1); the
// compute works on those rows only, the other rows are don't-care
#ifdef BCONV_LIVE
#define RD(src, dst, bytes) rd_live<bytes>(src, dst)
template <uint32_t BYTES>
inline void rd_live(uint64_t src, uint32_t dst) {
    constexpr uint32_t face = BYTES / 4, n = BCONV_LIVE * (BYTES / 64);
    noc_async_read(src, dst, n);
    noc_async_read(src + face, dst + face, n);
}
#else
#define RD(src, dst, bytes) noc_async_read(src, dst, bytes)
#endif


void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t s0_addr = get_arg_val<uint32_t>(1);
    const uint32_t s1_addr = get_arg_val<uint32_t>(2);
    const uint32_t s2_addr = get_arg_val<uint32_t>(3);
    const uint32_t w0_addr = get_arg_val<uint32_t>(4);
    const uint32_t w1_addr = get_arg_val<uint32_t>(5);
    const uint32_t w2_addr = get_arg_val<uint32_t>(6);
    const uint32_t wc_addr = get_arg_val<uint32_t>(7);
    const uint32_t t0 = get_arg_val<uint32_t>(8);
    const uint32_t nt = get_arg_val<uint32_t>(9);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_s = get_compile_time_arg_val(1);
    constexpr uint32_t cb_w = get_compile_time_arg_val(2);
    constexpr uint32_t x_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t F32 = 4096;
    constexpr auto x_args = TensorAccessorArgs<4>();
    constexpr auto st_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<st_args.next_compile_time_args_offset()>();
    const auto ax = TensorAccessor(x_args, x_addr, x_bytes);
    const auto as0 = TensorAccessor(st_args, s0_addr, F32);
    const auto as1 = TensorAccessor(st_args, s1_addr, F32);
    const auto as2 = TensorAccessor(st_args, s2_addr, F32);
    const auto aw0 = TensorAccessor(w_args, w0_addr, F32);
    const auto aw1 = TensorAccessor(w_args, w1_addr, F32);
    const auto aw2 = TensorAccessor(w_args, w2_addr, F32);
    const auto awc = TensorAccessor(w_args, wc_addr, F32);

#ifdef SEQ_R  // speculative verify rows (seqshift.hpp)
    const auto seq_a = TensorAccessor(TensorAccessorArgs<SEQ_CT>(), get_arg_val<uint32_t>(SEQ_ARG), 128);
    cb_reserve_back(SEQ_CB, 1);
    const uint32_t seq_scr = get_write_ptr(SEQ_CB);
    const auto seq_acc_rows = seq_acc(seq_a, seq_scr);
#endif
#ifndef BCONV_TAP_STRIDE
#define BCONV_TAP_STRIDE 0  // (> 0: the four taps are one tensor, tap j at tile t + j * stride)
#endif
#ifdef BCONV_DYN
    // dynamic taps (tt/bconv.py parts dyn=): the compute makes tap j = d @ e_j + sc_j; slots w0 / w1 / w2 = e / sc / d
    constexpr uint32_t cb_d = 8, cb_e = 9, cb_sc = 10;
    const auto asc = TensorAccessor(TensorAccessorArgs<BCONV_CT_SC>(), w1_addr, F32);
    const auto ad = TensorAccessor(TensorAccessorArgs<BCONV_CT_D>(), w2_addr, 2048);
    cb_reserve_back(cb_d, 1);
    noc_async_read(ad.get_noc_addr(0), get_write_ptr(cb_d), 2048);
    noc_async_read_barrier();
    cb_push_back(cb_d, 1);
#endif
    for (uint32_t t = t0; t < t0 + nt; ++t) {
        cb_reserve_back(cb_x, 1);
        cb_reserve_back(cb_s, 3);
#ifdef BCONV_DYN
        cb_reserve_back(cb_e, 4);
        cb_reserve_back(cb_sc, 4);
#else
        cb_reserve_back(cb_w, 4);
#endif
        RD(ax.get_noc_addr(t), get_write_ptr(cb_x), x_bytes);
        const uint32_t s = get_write_ptr(cb_s);
        RD(as0.get_noc_addr(t), s, F32);
        RD(as1.get_noc_addr(t), s + F32, F32);
        RD(as2.get_noc_addr(t), s + 2 * F32, F32);
#if defined(SEQ_R) && SEQ_ONE
        seq_pre(as0.get_noc_addr(t), as1.get_noc_addr(t), as2.get_noc_addr(t), 0, seq_scr, seq_acc_rows);
#endif
#ifdef BCONV_DYN
        const uint32_t e = get_write_ptr(cb_e), sc = get_write_ptr(cb_sc);
        const auto ae = TensorAccessor(w_args, w0_addr, 2048);
        for (uint32_t j = 0; j < 4; ++j) {
            noc_async_read(ae.get_noc_addr(t + j * BCONV_TAP_STRIDE), e + j * 2048, 2048);
            #ifdef BCONV_LIVE
            noc_async_read(asc.get_noc_addr(t + j * BCONV_TAP_STRIDE), sc + j * F32, 64);
            noc_async_read(asc.get_noc_addr(t + j * BCONV_TAP_STRIDE) + 1024, sc + j * F32 + 1024, 64);
#else
            noc_async_read(asc.get_noc_addr(t + j * BCONV_TAP_STRIDE), sc + j * F32, F32);
#endif
        }
        noc_async_read_barrier();
        // the static row (row 0) into rows 1 .. B-1, as the add's row broadcast
        for (uint32_t j = 0; j < 4; ++j) {
            const uint32_t base = sc + j * F32;
            for (uint32_t r = 1; r < BCONV_B; ++r) {
                const uint32_t o = (r >> 4) * 2048 + (r & 15) * 64;
                noc_async_read(get_noc_addr(base), base + o, 64);
                noc_async_read(get_noc_addr(base + 1024), base + o + 1024, 64);
            }
        }
        noc_async_read_barrier();
#else
        const uint32_t w = get_write_ptr(cb_w);
        RD(aw0.get_noc_addr(t), w, F32);
        RD(aw1.get_noc_addr(t + BCONV_TAP_STRIDE), w + F32, F32);
        RD(aw2.get_noc_addr(t + 2 * BCONV_TAP_STRIDE), w + 2 * F32, F32);
        RD(awc.get_noc_addr(t + 3 * BCONV_TAP_STRIDE), w + 3 * F32, F32);
        noc_async_read_barrier();
#endif
#ifdef SEQ_R
#if SEQ_ONE
        seq_fin<x_bytes == 4096 ? 4 : 2, 32, 1>(s, 0, get_write_ptr(cb_x), 0, seq_scr);
#else
        seq_shift<x_bytes == 4096 ? 4 : 2>(s, get_write_ptr(cb_x), seq_scr, seq_acc_rows);
#endif
#endif
        cb_push_back(cb_x, 1);
        cb_push_back(cb_s, 3);
#ifdef BCONV_DYN
        cb_push_back(cb_e, 4);
        cb_push_back(cb_sc, 4);
#else
        cb_push_back(cb_w, 4);
#endif
    }
}
