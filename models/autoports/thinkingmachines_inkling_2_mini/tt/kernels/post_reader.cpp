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
#ifdef SEQ_R
#include "seqshift.hpp"
#endif

// rows 0 .. T-1 of a tile (T <= 16: faces 0 / 1; e = element bytes) into a TH x 32 CB tile (faces min(TH, 16) x 16)
template <uint32_t T, uint32_t E, uint32_t TH>
inline void read_rows(uint64_t src, uint32_t dst) {
    constexpr uint32_t FACE = 256 * E;  // (rows 0 .. T-1 of a face are contiguous: one read per face)
    noc_async_read(src, dst, T * 16 * E);
    noc_async_read(src + FACE, dst + (TH < 16 ? TH : 16) * 16 * E, T * 16 * E);
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
    constexpr uint32_t T = POST_ROWS, TH = POST_TH;  // TH: the rows of the CB tiles of token rows (8 / 16 / 32)
    constexpr uint32_t F32 = TH * 128, BF = TH * 64;  // (a CB tile of token rows; tensor tiles are 32 x 32)
    auto rf = [&](uint64_t src, uint32_t dst) { read_rows<T, 4, TH>(src, dst); };
    auto rb = [&](uint64_t src, uint32_t dst) { read_rows<T, 2, TH>(src, dst); };
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
    const auto att = TensorAccessor(a_args, arg(), 2048);
    const auto xx = TensorAccessor(x_args, arg(), 2048);
    const auto s0 = TensorAccessor(s_args, arg(), 4096);
    const auto s1 = TensorAccessor(s_args, arg(), 4096);
    const auto s2 = TensorAccessor(s_args, arg(), 4096);
    const auto w0 = TensorAccessor(w_args, arg(), 4096);
    const auto w1 = TensorAccessor(w_args, arg(), 4096);
    const auto w2 = TensorAccessor(w_args, arg(), 4096);
    const auto wc = TensorAccessor(w_args, arg(), 4096);
    const auto q = TensorAccessor(w_args, arg(), 4096);
    const auto ones = TensorAccessor(w_args, arg(), 2048);
    const auto nw = TensorAccessor(w_args, arg(), 4096);
    const auto sel = TensorAccessor(w_args, arg(), 2048);

#ifdef SEQ_R  // speculative verify rows (seqshift.hpp)
    const auto seq_a = TensorAccessor(TensorAccessorArgs<SEQ_CT>(), get_common_arg_val<uint32_t>(SEQ_ARG), 128);
    cb_reserve_back(SEQ_CB, 1);
    const uint32_t seq_scr = get_write_ptr(SEQ_CB);
    const auto seq_acc_rows = seq_acc(seq_a, seq_scr);
#endif
    // every conv input of this core's tiles at once (one barrier), pushed tile by tile
    cb_reserve_back(cb_att, PER);
    cb_reserve_back(cb_s, 3 * PER);
    cb_reserve_back(cb_w, 4 * PER);
    if constexpr (ADD) {
        cb_reserve_back(cb_x, PER);
    }
    for (uint32_t i = 0; i < PER; ++i) {
        const uint32_t t = t0 + i;
#if POST_ATT_PACKED
        for (uint32_t f = 0; f < 2; ++f) {  // chunks (t, f, rows 0 .. T-1) of the packed all-reduce result (tt/ccl.py)
            const uint32_t ch = (t * 2 + f) * T, n = 64 - (ch & 63) < T ? 64 - (ch & 63) : T;  // (a run may cross a page)
            const uint32_t d = get_write_ptr(cb_att) + i * BF + f * (TH < 16 ? TH : 16) * 32;
            noc_async_read(att.get_noc_addr(ch >> 6) + (ch & 63) * 32, d, n * 32);
            if (n < T) {
                noc_async_read(att.get_noc_addr((ch >> 6) + 1), d + n * 32, (T - n) * 32);
            }
        }
#else
        rb(att.get_noc_addr(t), get_write_ptr(cb_att) + i * BF);
#endif
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
#if defined(SEQ_R) && SEQ_ONE
        if constexpr (seq_fits<PER>()) {
            seq_pre(s0.get_noc_addr(t), s1.get_noc_addr(t), s2.get_noc_addr(t), i, seq_scr, seq_acc_rows);
        }
#endif
    }
    noc_async_read_barrier();
#ifdef SEQ_R
    if constexpr (SEQ_ONE && seq_fits<PER>()) {
        seq_fin<2, TH, PER>(get_write_ptr(cb_s), 3 * F32, get_write_ptr(cb_att), BF, seq_scr);
    } else {
        for (uint32_t i = 0; i < PER; ++i) {
            seq_shift<2, TH>(get_write_ptr(cb_s) + i * 3 * F32, get_write_ptr(cb_att) + i * BF, seq_scr, seq_acc_rows);
        }
    }
#endif
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
    noc_async_read(ones.get_noc_addr(0), get_write_ptr(cb_ones), 2048);
    cb_reserve_back(cb_sel, NQ);
    for (uint32_t k = 0; k < NQ; ++k) {  // (row 0 of the 0 / 1 tiles is the only nonzero row: TH rows of them)
        if constexpr (TH == 32) {
            noc_async_read(sel.get_noc_addr(k), get_write_ptr(cb_sel) + k * BF, BF);
        } else {
            read_rows<TH, 2, TH>(sel.get_noc_addr(k), get_write_ptr(cb_sel) + k * BF);
        }
    }
#if POST_RD_STAGED  // (the round-1 inputs pushed first, then the mix's, then the router's)
    noc_async_read_barrier();
    cb_push_back(cb_q, PER);
    cb_push_back(cb_ones, 1);
    cb_push_back(cb_sel, NQ);
#endif
    cb_reserve_back(cb_d, NB);
    cb_reserve_back(cb_blk, NB * PER);
    const uint32_t a_blk = a;
    for (uint32_t b = 0; b < NB; ++b) {
        const auto bt = TensorAccessor(b_args, get_common_arg_val<uint32_t>(a_blk + 3 * b), 4096);
        const auto dl = TensorAccessor(b_args, get_common_arg_val<uint32_t>(a_blk + 3 * b + 1), 4096);
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
#if POST_RD_STAGED
    noc_async_read_barrier();
    cb_push_back(cb_d, NB);
    cb_push_back(cb_blk, NB * PER);
    if constexpr (NORM) {
        cb_push_back(cb_nw, PER);
    }
#endif
#if POST_RT
    {  // the router weight's bf16 hi / lo rows of this core's tiles: [half][t][e]
        constexpr uint32_t ET = POST_RT;
        cb_reserve_back(POST_CB_RW, 2 * PER * ET);
        const uint32_t rw = get_write_ptr(POST_CB_RW);
        for (uint32_t half = 0; half < 2; ++half) {
            const auto ar = TensorAccessor(w_args, get_common_arg_val<uint32_t>(a_blk + 3 * NB + half), 2048);
            for (uint32_t i = 0; i < PER; ++i) {
                for (uint32_t e = 0; e < ET; ++e) {
                    noc_async_read(ar.get_noc_addr((t0 + i) * ET + e), rw + ((half * PER + i) * ET + e) * 2048, 2048);
                }
            }
        }
    }
#endif
    noc_async_read_barrier();
#if POST_RT
    cb_push_back(POST_CB_RW, 2 * PER * POST_RT);
#endif
#if !POST_RD_STAGED
    cb_push_back(cb_q, PER);
    cb_push_back(cb_ones, 1);
    cb_push_back(cb_sel, NQ);
    cb_push_back(cb_d, NB);
    cb_push_back(cb_blk, NB * PER);
    if constexpr (NORM) {
        cb_push_back(cb_nw, PER);
    }
#endif
}
