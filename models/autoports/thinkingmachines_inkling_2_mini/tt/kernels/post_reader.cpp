// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fused decode post-attention section (tt/post.py), reader: this core's PER hidden tiles [t0, t0 + PER).
// Order: conv inputs per tile (att, x, 3 states, 4 taps), then the query tiles, ones, the block logits, the block
// tiles and the norm weight tiles.
//
// runtime args: t0, att, x, s0, s1, s2, w0, w1, w2, wc, q, ones, nw, then NB x (block addr, logits addr, logits tile)

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// rows 0 .. T-1 of a tile (T <= 16: faces 0 / 1; e = element bytes): 2 T half-row reads
template <uint32_t T, uint32_t E>
inline void read_rows(uint64_t src, uint32_t dst) {
    constexpr uint32_t HALF = 16 * E, FACE = 256 * E;
    for (uint32_t r = 0; r < T; ++r) {
        noc_async_read(src + r * HALF, dst + r * HALF, HALF);
        noc_async_read(src + FACE + r * HALF, dst + FACE + r * HALF, HALF);
    }
}

void kernel_main() {
    constexpr uint32_t cb_att = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x = get_compile_time_arg_val(1);
    constexpr uint32_t cb_s = get_compile_time_arg_val(2);
    constexpr uint32_t cb_w = get_compile_time_arg_val(3);
    constexpr uint32_t cb_q = get_compile_time_arg_val(4);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(5);
    constexpr uint32_t cb_d = get_compile_time_arg_val(6);
    constexpr uint32_t cb_blk = get_compile_time_arg_val(7);
    constexpr uint32_t cb_nw = get_compile_time_arg_val(8);
    constexpr uint32_t PER = get_compile_time_arg_val(9);
    constexpr uint32_t NB = get_compile_time_arg_val(10);
    constexpr uint32_t ADD = get_compile_time_arg_val(11);
    constexpr uint32_t NORM = get_compile_time_arg_val(12);
    constexpr uint32_t cb_sel = get_compile_time_arg_val(13);
    constexpr uint32_t NQ = get_compile_time_arg_val(14);
    constexpr uint32_t F32 = 4096, BF = 2048;
    constexpr uint32_t T = POST_ROWS;
    auto rf = [&](uint64_t src, uint32_t dst) { read_rows<T, 4>(src, dst); };
    auto rb = [&](uint64_t src, uint32_t dst) { read_rows<T, 2>(src, dst); };
    constexpr auto a_args = TensorAccessorArgs<15>();  // att
    constexpr auto x_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();  // x
    constexpr auto s_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();  // states
    constexpr auto w_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();  // taps, q, ones, norm w
    constexpr auto b_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();  // blocks, block logits

    // this core's slot from its logical coordinates (every other argument is common to all cores)
    const uint32_t c = (uint32_t)get_absolute_logical_y() * POST_CX + (uint32_t)get_absolute_logical_x();
    const uint32_t t0 = c * PER;
    uint32_t a = 0;
    auto arg = [&]() { return get_common_arg_val<uint32_t>(a++); };
    const auto att = TensorAccessor(a_args, arg(), BF);
    const auto xx = TensorAccessor(x_args, arg(), BF);
    const auto s0 = TensorAccessor(s_args, arg(), F32);
    const auto s1 = TensorAccessor(s_args, arg(), F32);
    const auto s2 = TensorAccessor(s_args, arg(), F32);
    const auto w0 = TensorAccessor(w_args, arg(), F32);
    const auto w1 = TensorAccessor(w_args, arg(), F32);
    const auto w2 = TensorAccessor(w_args, arg(), F32);
    const auto wc = TensorAccessor(w_args, arg(), F32);
    const auto q = TensorAccessor(w_args, arg(), F32);
    const auto ones = TensorAccessor(w_args, arg(), BF);
    const auto nw = TensorAccessor(w_args, arg(), F32);
    const auto sel = TensorAccessor(w_args, arg(), BF);

    // every conv input of this core's tiles at once (one barrier), pushed tile by tile
    cb_reserve_back(cb_att, PER);
    cb_reserve_back(cb_s, 3 * PER);
    cb_reserve_back(cb_w, 4 * PER);
    if constexpr (ADD) {
        cb_reserve_back(cb_x, PER);
    }
    for (uint32_t i = 0; i < PER; ++i) {
        const uint32_t t = t0 + i;
        rb(att.get_noc_addr(t), get_write_ptr(cb_att) + i * BF);
        if constexpr (ADD) {
            rb(xx.get_noc_addr(t), get_write_ptr(cb_x) + i * BF);
        }
        const uint32_t s = get_write_ptr(cb_s) + i * 3 * F32;
        rf(s0.get_noc_addr(t), s);
        rf(s1.get_noc_addr(t), s + F32);
        rf(s2.get_noc_addr(t), s + 2 * F32);
        const uint32_t w = get_write_ptr(cb_w) + i * 4 * F32;
        rf(w0.get_noc_addr(t), w);
        rf(w1.get_noc_addr(t), w + F32);
        rf(w2.get_noc_addr(t), w + 2 * F32);
        rf(wc.get_noc_addr(t), w + 3 * F32);
    }
    noc_async_read_barrier();
    for (uint32_t i = 0; i < PER; ++i) {
        cb_push_back(cb_att, 1);
        if constexpr (ADD) {
            cb_push_back(cb_x, 1);
        }
        cb_push_back(cb_s, 3);
        cb_push_back(cb_w, 4);
    }
    cb_reserve_back(cb_q, PER);
    for (uint32_t i = 0; i < PER; ++i) {
        rf(q.get_noc_addr(t0 + i), get_write_ptr(cb_q) + i * F32);
    }
    cb_reserve_back(cb_ones, 1);
    noc_async_read(ones.get_noc_addr(0), get_write_ptr(cb_ones), BF);
    cb_reserve_back(cb_sel, NQ);
    for (uint32_t k = 0; k < NQ; ++k) {
        noc_async_read(sel.get_noc_addr(k), get_write_ptr(cb_sel) + k * BF, BF);
    }
    cb_reserve_back(cb_d, NB);
    cb_reserve_back(cb_blk, NB * PER);
    const uint32_t a_blk = a;
    for (uint32_t b = 0; b < NB; ++b) {
        const auto bt = TensorAccessor(b_args, get_common_arg_val<uint32_t>(a_blk + 3 * b), F32);
        const auto dl = TensorAccessor(b_args, get_common_arg_val<uint32_t>(a_blk + 3 * b + 1), F32);
        rf(dl.get_noc_addr(get_common_arg_val<uint32_t>(a_blk + 3 * b + 2)), get_write_ptr(cb_d) + b * F32);
        for (uint32_t i = 0; i < PER; ++i) {
            rf(bt.get_noc_addr(t0 + i), get_write_ptr(cb_blk) + (b * PER + i) * F32);
        }
    }
    if constexpr (NORM) {
        cb_reserve_back(cb_nw, PER);
        for (uint32_t i = 0; i < PER; ++i) {
            rf(nw.get_noc_addr(t0 + i), get_write_ptr(cb_nw) + i * F32);
        }
    }
#if POST_RT
    {  // the router weight's bf16 hi / lo rows of this core's tiles: [half][t][e]
        constexpr uint32_t ET = POST_RT;
        cb_reserve_back(POST_CB_RW, 2 * PER * ET);
        const uint32_t rw = get_write_ptr(POST_CB_RW);
        for (uint32_t half = 0; half < 2; ++half) {
            const auto ar = TensorAccessor(w_args, get_common_arg_val<uint32_t>(a_blk + 3 * NB + half), BF);
            for (uint32_t i = 0; i < PER; ++i) {
                for (uint32_t e = 0; e < ET; ++e) {
                    noc_async_read(ar.get_noc_addr((t0 + i) * ET + e), rw + ((half * PER + i) * ET + e) * BF, BF);
                }
            }
        }
    }
#endif
    noc_async_read_barrier();
#if POST_RT
    cb_push_back(POST_CB_RW, 2 * PER * POST_RT);
#endif
    cb_push_back(cb_q, PER);
    cb_push_back(cb_ones, 1);
    cb_push_back(cb_sel, NQ);
    cb_push_back(cb_d, NB);
    cb_push_back(cb_blk, NB * PER);
    if constexpr (NORM) {
        cb_push_back(cb_nw, PER);
    }
}
