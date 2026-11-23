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
    for (uint32_t t = t0; t < t0 + nt; ++t) {
        cb_reserve_back(cb_x, 1);
        cb_reserve_back(cb_s, 3);
        cb_reserve_back(cb_w, 4);
        noc_async_read(ax.get_noc_addr(t), get_write_ptr(cb_x), x_bytes);
        const uint32_t s = get_write_ptr(cb_s);
        noc_async_read(as0.get_noc_addr(t), s, F32);
        noc_async_read(as1.get_noc_addr(t), s + F32, F32);
        noc_async_read(as2.get_noc_addr(t), s + 2 * F32, F32);
        const uint32_t w = get_write_ptr(cb_w);
#ifndef BCONV_TAP_STRIDE
#define BCONV_TAP_STRIDE 0  // (> 0: the four taps are one tensor, tap j at tile t + j * stride)
#endif
        noc_async_read(aw0.get_noc_addr(t), w, F32);
        noc_async_read(aw1.get_noc_addr(t + BCONV_TAP_STRIDE), w + F32, F32);
        noc_async_read(aw2.get_noc_addr(t + 2 * BCONV_TAP_STRIDE), w + 2 * F32, F32);
        noc_async_read(awc.get_noc_addr(t + 3 * BCONV_TAP_STRIDE), w + 3 * F32, F32);
        noc_async_read_barrier();
#ifdef SEQ_R
        seq_shift<x_bytes == 4096 ? 4 : 2>(s, get_write_ptr(cb_x), seq_scr, seq_acc_rows);
#endif
        cb_push_back(cb_x, 1);
        cb_push_back(cb_s, 3);
        cb_push_back(cb_w, 4);
    }
}
