// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fused decode pre-attention section (tt/pre.py), reader: this core's PER hidden tiles, live rows only.
// TAIL: mo (bf16), the previous residual (fp32), the MoE-output conv's 3 states + 4 taps; else x (bf16). Then the NQ
// query rows, ones, the selection tiles, the block logits (NB x NQ), the blocks, the NQ norm weight rows.
//
// common args: x_or_mo, resid, s0, s1, s2, w0, w1, w2, wc, ones, sel, q[NQ], nw[NQ], then NB x (block, logits, tile)

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#ifdef SEQ_R
#include "seqshift.hpp"
#endif

template <uint32_t T, uint32_t E>
inline void read_rows(uint64_t src, uint32_t dst) {
    constexpr uint32_t FACE = 256 * E;  // (rows 0 .. T-1 of a face are contiguous: one read per face)
    noc_async_read(src, dst, T * 16 * E);
    noc_async_read(src + FACE, dst + FACE, T * 16 * E);
}

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);   // mo | x (bf16)
    constexpr uint32_t cb_rp = get_compile_time_arg_val(1);   // previous residual (fp32)
    constexpr uint32_t cb_s = get_compile_time_arg_val(2);
    constexpr uint32_t cb_w = get_compile_time_arg_val(3);
    constexpr uint32_t cb_q = get_compile_time_arg_val(4);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(5);
    constexpr uint32_t cb_d = get_compile_time_arg_val(6);
    constexpr uint32_t cb_blk = get_compile_time_arg_val(7);
    constexpr uint32_t cb_nw = get_compile_time_arg_val(8);
    constexpr uint32_t cb_sel = get_compile_time_arg_val(9);
    constexpr uint32_t PER = get_compile_time_arg_val(10);
    constexpr uint32_t NB = get_compile_time_arg_val(11);
    constexpr uint32_t NQ = get_compile_time_arg_val(12);
    constexpr uint32_t TAIL = get_compile_time_arg_val(13);
    constexpr uint32_t NORM = get_compile_time_arg_val(14);
    constexpr uint32_t NSEL = get_compile_time_arg_val(15);
    constexpr uint32_t F32 = 4096, BF = 2048;
    constexpr uint32_t T = PRE_ROWS;
    constexpr auto i_args = TensorAccessorArgs<16>();                                     // mo | x
    constexpr auto r_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();  // residual
    constexpr auto s_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();  // states
    constexpr auto w_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();  // taps, ones, sel, q, nw
    constexpr auto b_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();  // blocks, logits
    auto rf = [&](uint64_t src, uint32_t dst) { read_rows<T, 4>(src, dst); };
    auto rb = [&](uint64_t src, uint32_t dst) { read_rows<T, 2>(src, dst); };

    const uint32_t c = (uint32_t)get_absolute_logical_y() * PRE_CX + (uint32_t)get_absolute_logical_x();
    const uint32_t t0 = c * PER;
    auto ca = [](uint32_t i) { return get_common_arg_val<uint32_t>(i); };
    const auto ain = TensorAccessor(i_args, ca(0), BF);
    const auto arp = TensorAccessor(r_args, ca(1), F32);

#ifdef SEQ_R  // speculative verify rows (seqshift.hpp)
    const auto seq_a = TensorAccessor(TensorAccessorArgs<SEQ_CT>(), ca(SEQ_ARG), 128);
    cb_reserve_back(SEQ_CB, 1);
    const uint32_t seq_scr = get_write_ptr(SEQ_CB);
    const auto seq_acc_rows = seq_acc(seq_a, seq_scr);
#endif
    cb_reserve_back(cb_in, PER);
    if constexpr (TAIL) {
        cb_reserve_back(cb_rp, PER);
        cb_reserve_back(cb_s, 3 * PER);
        cb_reserve_back(cb_w, 4 * PER);
    }
    for (uint32_t i = 0; i < PER; ++i) {
        const uint32_t t = t0 + i;
#if PRE_IN_PACKED
        for (uint32_t r = 0; r < T; ++r) {  // chunk (t, r, f) of the packed all-reduce result (tt/ccl.py)
            for (uint32_t f = 0; f < 2; ++f) {
                const uint32_t ch = (t * T + r) * 2 + f;
                noc_async_read(ain.get_noc_addr(ch >> 6) + (ch & 63) * 32,
                               get_write_ptr(cb_in) + i * BF + f * 512 + r * 32, 32);
            }
        }
#else
        rb(ain.get_noc_addr(t), get_write_ptr(cb_in) + i * BF);
#endif
        if constexpr (TAIL) {
            rf(arp.get_noc_addr(t), get_write_ptr(cb_rp) + i * F32);
            for (uint32_t j = 0; j < 3; ++j) {
                const auto as = TensorAccessor(s_args, ca(2 + j), F32);
                rf(as.get_noc_addr(t), get_write_ptr(cb_s) + (3 * i + j) * F32);
            }
            for (uint32_t j = 0; j < 4; ++j) {
                const auto aw = TensorAccessor(w_args, ca(5 + j), F32);
                rf(aw.get_noc_addr(t), get_write_ptr(cb_w) + (4 * i + j) * F32);
            }
        }
    }
    noc_async_read_barrier();
#ifdef SEQ_R
    for (uint32_t i = 0; i < PER; ++i) {
        seq_shift<2>(get_write_ptr(cb_s) + 3 * i * F32, get_write_ptr(cb_in) + i * BF, seq_scr, seq_acc_rows);
    }
#endif
    for (uint32_t i = 0; i < PER; ++i) {
        cb_push_back(cb_in, 1);
        if constexpr (TAIL) {
            cb_push_back(cb_rp, 1);
            cb_push_back(cb_s, 3);
            cb_push_back(cb_w, 4);
        }
    }

    const auto ones = TensorAccessor(w_args, ca(9), BF);
    const auto sel = TensorAccessor(w_args, ca(10), BF);
    cb_reserve_back(cb_ones, 1);
    noc_async_read(ones.get_noc_addr(0), get_write_ptr(cb_ones), BF);
    cb_reserve_back(cb_sel, NSEL);
    for (uint32_t k = 0; k < NSEL; ++k) {
        noc_async_read(sel.get_noc_addr(k), get_write_ptr(cb_sel) + k * BF, BF);
    }
    cb_reserve_back(cb_q, NQ * PER);  // [q][tile]
    for (uint32_t q = 0; q < NQ; ++q) {
        const auto aq = TensorAccessor(w_args, ca(11 + q), F32);
        for (uint32_t i = 0; i < PER; ++i) {
            rf(aq.get_noc_addr(t0 + i), get_write_ptr(cb_q) + (q * PER + i) * F32);
        }
    }
    constexpr uint32_t ab = 11 + 2 * NQ;
    cb_reserve_back(cb_d, NB * NQ);  // [b][q]
    cb_reserve_back(cb_blk, NB * PER);
    for (uint32_t b = 0; b < NB; ++b) {
        const auto bt = TensorAccessor(b_args, ca(ab + 3 * b), F32);
        const auto dl = TensorAccessor(b_args, ca(ab + 3 * b + 1), F32);
        const uint32_t off = ca(ab + 3 * b + 2);
        for (uint32_t q = 0; q < NQ; ++q) {
            rf(dl.get_noc_addr(off + q), get_write_ptr(cb_d) + (b * NQ + q) * F32);
        }
        for (uint32_t i = 0; i < PER; ++i) {
            rf(bt.get_noc_addr(t0 + i), get_write_ptr(cb_blk) + (b * PER + i) * F32);
        }
    }
    if constexpr (NORM) {
        cb_reserve_back(cb_nw, NQ * PER);
        for (uint32_t q = 0; q < NQ; ++q) {
            const auto an = TensorAccessor(w_args, ca(11 + NQ + q), F32);
            for (uint32_t i = 0; i < PER; ++i) {
                rf(an.get_noc_addr(t0 + i), get_write_ptr(cb_nw) + (q * PER + i) * F32);
            }
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_ones, 1);
    cb_push_back(cb_sel, NSEL);
    cb_push_back(cb_q, NQ * PER);
    cb_push_back(cb_d, NB * NQ);
    cb_push_back(cb_blk, NB * PER);
    if constexpr (NORM) {
        cb_push_back(cb_nw, NQ * PER);
    }
}
