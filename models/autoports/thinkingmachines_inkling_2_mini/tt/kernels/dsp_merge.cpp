// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// DSpark drafter, vocab-sharded Markov step merge (tt/drafter.py _step, NS > 1; one core, the same program on every
// die). Input: the NS slices' local results gathered ([1, 1, 1, NS * 2 K] uint32 TILE, row 0: slice s's K (key, local id)
// pairs, keys in the order-preserving form of kernels/dsp_step_writer.cpp). Ids become global (+ s * VS); padded ids
// (>= V) and empty pairs drop out. Greedy: the max key (ties: lowest id). SAMPLE: the top K over all slices
// (descending, ties: lowest id), softcapped through the host table, ids / logits to row KS of cand_id / cand_lg, token
// = argmax_j (logit_j + tg_j). Then the token -> prev, w1[token] -> row 0 of the next step's wp.
//
// runtime args: [g_addr, prev_addr, w1_addr, wp_addr (, gum_addr, cid_addr, clg_addr, tab_addr)]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "dsp_w1row.hpp"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t NS = get_compile_time_arg_val(1);
    constexpr uint32_t K = get_compile_time_arg_val(2);
    constexpr uint32_t VS = get_compile_time_arg_val(3);
    constexpr uint32_t V = get_compile_time_arg_val(4);
    constexpr uint32_t KS = get_compile_time_arg_val(5);
    constexpr uint32_t NTAB = get_compile_time_arg_val(6);
    constexpr auto g_args = TensorAccessorArgs<7>();
    constexpr auto p_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto wp_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr uint32_t NV = NS * 2 * K;  // gathered values
    constexpr uint32_t NT = NV / 32;     // gathered tiles

    const uint32_t l1 = get_write_ptr(cb);  // [0, 4 NV): values; then scratch
    const uint32_t scr = l1 + ((4 * NV + 63) / 64) * 64;
    const auto s_g = TensorAccessor(g_args, get_arg_val<uint32_t>(0), 4096);
    for (uint32_t t = 0; t < NT; ++t) {
        const uint64_t a = s_g.get_noc_addr(t);
        noc_async_read(a, l1 + t * 128, 64);
        noc_async_read(a + 1024, l1 + t * 128 + 64, 64);
    }
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* v = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    uint32_t key[NS * K], gid[NS * K];
    for (uint32_t s = 0; s < NS; ++s) {
        for (uint32_t j = 0; j < K; ++j) {
            const uint32_t k = v[s * 2 * K + 2 * j], id = v[s * 2 * K + 2 * j + 1];
            const uint32_t g = id + s * VS;
            const bool ok = id != 0xFFFFFFFFu && g < V;
            key[s * K + j] = ok ? k : 0;
            gid[s * K + j] = ok ? g : 0xFFFFFFFFu;
        }
    }
#ifdef SAMPLE
    const uint32_t tg_l1 = scr, tab = scr + 4 * K, out = tab + ((4 * NTAB + 63) / 64) * 64, wscr = out + 8 * K + 64;
    constexpr auto gu_args = TensorAccessorArgs<wp_args.next_compile_time_args_offset()>();
    constexpr auto ci_args = TensorAccessorArgs<gu_args.next_compile_time_args_offset()>();
    constexpr auto cl_args = TensorAccessorArgs<ci_args.next_compile_time_args_offset()>();
    constexpr auto t_args = TensorAccessorArgs<cl_args.next_compile_time_args_offset()>();
    noc_async_read(TensorAccessor(gu_args, get_arg_val<uint32_t>(4), 4 * K).get_noc_addr(KS), tg_l1, 4 * K);
    noc_async_read(TensorAccessor(t_args, get_arg_val<uint32_t>(7), 4 * NTAB).get_noc_addr(0), tab, 4 * NTAB);
    // top K by repeated selection (a dropped pad can sit inside a slice's sorted list)
    uint32_t tk[K], ti[K];
    for (uint32_t j = 0; j < K; ++j) {
        uint32_t bq = 0xFFFFFFFFu, bk = 0, bi = 0xFFFFFFFFu;
        for (uint32_t q = 0; q < NS * K; ++q) {
            if (gid[q] == 0xFFFFFFFFu) {
                continue;
            }
            if (bq == 0xFFFFFFFFu || key[q] > bk || (key[q] == bk && gid[q] < bi)) {
                bq = q;
                bk = key[q];
                bi = gid[q];
            }
        }
        if (bq != 0xFFFFFFFFu) {
            gid[bq] = 0xFFFFFFFFu;
        }
        tk[j] = (bk & 0x80000000u) ? (bk & 0x7FFFFFFFu) : ~bk;  // the FP32 bits back
        ti[j] = bi;
    }
    noc_async_read_barrier();
    volatile tt_l1_ptr float* tg = reinterpret_cast<volatile tt_l1_ptr float*>(tg_l1);
    volatile tt_l1_ptr float* ct = reinterpret_cast<volatile tt_l1_ptr float*>(tab);
    volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out);
    uint32_t best = 0;
    float bv = 0.0f;
    for (uint32_t j = 0; j < K; ++j) {
        union {
            uint32_t u;
            float f;
        } cv;
        cv.u = tk[j];
        float x = (cv.f + CAP_L) * CAP_INV_H;
        x = x < 0.0f ? 0.0f : (x > (float)(NTAB - 2) ? (float)(NTAB - 2) : x);
        const uint32_t i = (uint32_t)x;
        const float fr = x - (float)i;
        cv.f = ct[i] + fr * (ct[i + 1] - ct[i]);
        o[j] = ti[j];
        o[K + j] = cv.u;
        const float val = cv.f + tg[j];
        if (j == 0 || val > bv) {
            bv = val;
            best = j;
        }
    }
    noc_async_write(out, TensorAccessor(ci_args, get_arg_val<uint32_t>(5), 4 * K).get_noc_addr(KS), 4 * K);
    noc_async_write(out + 4 * K, TensorAccessor(cl_args, get_arg_val<uint32_t>(6), 4 * K).get_noc_addr(KS), 4 * K);
    noc_async_write_barrier();
    const uint32_t tok = ti[best];
#else
    const uint32_t out = scr, wscr = scr + 64;
    volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out);
    uint32_t bk = 0, tok = 0xFFFFFFFFu;
    for (uint32_t q = 0; q < NS * K; ++q) {
        if (gid[q] != 0xFFFFFFFFu && (tok == 0xFFFFFFFFu || key[q] > bk || (key[q] == bk && gid[q] < tok))) {
            bk = key[q];
            tok = gid[q];
        }
    }
#endif
    o[0] = tok;
    noc_async_write(out, TensorAccessor(p_args, get_arg_val<uint32_t>(1), 4).get_noc_addr(0), 4);
    noc_async_write_barrier();
    w1_row(TensorAccessor(w_args, get_arg_val<uint32_t>(2), 512), TensorAccessor(wp_args, get_arg_val<uint32_t>(3), 2048), tok,
           ((wscr + 63) / 64) * 64);
}
