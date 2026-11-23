// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// DSpark Markov step, activations + argmax (BRISC; tt/drafter.py MarkovStep). Row 0 of the kt - 1 bf16 latent tiles
// into cb_x plus a ones tile (x[0, 256] = 1: the fold bias row), this core's base logits (row `row` of the compact
// [1, 8, NT, 32] FP32 tensor) as one FP32 tile per column (row 0 live) into cb_base; then row 0 of each output tile is
// scanned for this core's max (integer order of the FP32 bits; ascending vocab ids, strict >: lowest id on ties;
// ids >= V skipped). Every core sends (key, id) to the reducer's cb_cand slot and bumps its semaphore; the reducer
// picks the global max (ties: lowest id) and writes the token to `prev` (uint32 [1, 1] ROW_MAJOR).
//
// SAMPLE (sampled drafts): each core keeps its top TOPK (key, id) instead of the max; the reducer merges them into the
// step's top TOPK (descending, ties: lowest id), softcaps them (c tanh(x / c) interpolated in a host-made FP32 table on
// the grid -CAP_L + i / CAP_INV_H, i < NTAB: error ~1e-5), writes their ids / capped logits to row KS of cand_id /
// cand_lg ([d, TOPK] uint32 / FP32 ROW_MAJOR) and draws the token as argmax_j (logit_j + tg_j), tg = row KS of the
// host-written T * Gumbel noise ([d, TOPK] FP32; all zero = the top-1).
//
// The reducer also writes w1[token] as row 0 of the next step's input wp ([1, 1, 1, 256] bf16 TILE).
// LOCAL_OUT (vocab-sharded drafter: this program sees one vocab slice): the reducer only writes its result, the best
// (key, local id) pair (greedy) or its top K pairs (SAMPLE; keys in the order-preserving key form), to row 0 of `loc`
// ([1, 1, 1, 2 TOPK] uint32 TILE; arg 6 = its address, unused pairs key 0); kernels/dsp_merge.cpp picks the token.
//
// runtime args: [x_addr, base_addr, prev_addr, col0, ncols, slot, w1_addr, wp_addr (, gum, cid, clg, tab addrs)]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "dsp_w1row.hpp"

static inline uint32_t fkey(uint32_t u) { return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }

#ifdef LOCAL_OUT
// pairs (k[j], i[j]), j < n, then (0, ~0) up to LOCAL_OUT pairs -> row 0 of loc (2 LOCAL_OUT uint32: tiles of 32)
template <typename A>
inline void write_loc(const A& s_l, const uint32_t* k, const uint32_t* i, uint32_t n, uint32_t st) {
    volatile tt_l1_ptr uint32_t* v = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(st);
    for (uint32_t p = 0; p < 2 * LOCAL_OUT; ++p) {
        const uint32_t j = p >> 1;
        const uint32_t x = (p & 1) ? (j < n ? i[j] : 0xFFFFFFFFu) : (j < n ? k[j] : 0u);
        const uint32_t t = p >> 5, c = p & 31;
        v[t * 32 + c] = x;  // tile t's staging: words [0, 16) = face 0 row 0, [16, 32) = face 1 row 0
    }
    for (uint32_t t = 0; t < (2 * LOCAL_OUT + 31) / 32; ++t) {
        const uint64_t d = s_l.get_noc_addr(t);
        noc_async_write(st + t * 128, d, 64);
        noc_async_write(st + t * 128 + 64, d + 1024, 64);
    }
    noc_async_write_barrier();
}
#endif

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t base_addr = get_arg_val<uint32_t>(1);
    const uint32_t prev_addr = get_arg_val<uint32_t>(2);
    const uint32_t col0 = get_arg_val<uint32_t>(3);
    const uint32_t ncols = get_arg_val<uint32_t>(4);
    const uint32_t slot = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t cb_base = get_compile_time_arg_val(2);
    constexpr uint32_t cb_bseg = get_compile_time_arg_val(3);
    constexpr uint32_t cb_cand = get_compile_time_arg_val(4);
    constexpr uint32_t kt = get_compile_time_arg_val(5);
    constexpr uint32_t V = get_compile_time_arg_val(6);
    constexpr uint32_t NT = get_compile_time_arg_val(7);
    constexpr uint32_t row = get_compile_time_arg_val(8);
    constexpr uint32_t ncores = get_compile_time_arg_val(9);
    constexpr uint32_t red_x = get_compile_time_arg_val(10);
    constexpr uint32_t red_y = get_compile_time_arg_val(11);
    constexpr auto x_args = TensorAccessorArgs<12>();
    constexpr auto b_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto wp_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
#ifdef SAMPLE
    constexpr uint32_t K = TOPK;
    constexpr auto g_args = TensorAccessorArgs<wp_args.next_compile_time_args_offset()>();
    constexpr auto ci_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto cl_args = TensorAccessorArgs<ci_args.next_compile_time_args_offset()>();
    constexpr auto t_args = TensorAccessorArgs<cl_args.next_compile_time_args_offset()>();
    constexpr uint32_t KS = get_compile_time_arg_val(t_args.next_compile_time_args_offset());
    constexpr uint32_t NTAB = get_compile_time_arg_val(t_args.next_compile_time_args_offset() + 1);
    uint32_t tk[K], ti[K], cnt = 0;
#endif

    const uint32_t sem = get_semaphore(0);
    const uint32_t cand = get_write_ptr(cb_cand);
    uint32_t best_key = 0, best_id = 0xFFFFFFFFu;

    if (ncols > 0) {
        cb_reserve_back(cb_x, kt);
        const uint32_t xl1 = get_write_ptr(cb_x);
        const auto s_x = TensorAccessor(x_args, x_addr, 2048);
        for (uint32_t k = 0; k + 1 < kt; ++k) {
            const uint64_t src = s_x.get_noc_addr(k);
            noc_async_read(src, xl1 + k * 2048, 32);
            noc_async_read(src + 512, xl1 + k * 2048 + 512, 32);
        }
        volatile tt_l1_ptr uint32_t* ones = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xl1 + (kt - 1) * 2048);
        for (uint32_t i = 0; i < 8; ++i) {
            ones[i] = 0;
            ones[128 + i] = 0;
        }
        ones[0] = 0x3F80u;  // bf16 1.0 at [0, 0]
        const uint32_t bseg = get_write_ptr(cb_bseg);
        const auto s_b = TensorAccessor(b_args, base_addr, 128);
        for (uint32_t i = 0; i < ncols; ++i) {
            noc_async_read(s_b.get_noc_addr(row * NT + col0 + i), bseg + i * 128, 128);
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, kt);

        for (uint32_t i = 0; i < ncols; ++i) {
            cb_reserve_back(cb_base, 1);
            volatile tt_l1_ptr uint32_t* bt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_base));
            volatile tt_l1_ptr uint32_t* bs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bseg + i * 128);
            for (uint32_t j = 0; j < 16; ++j) {
                bt[j] = bs[j];
                bt[256 + j] = bs[16 + j];
            }
            cb_push_back(cb_base, 1);

            cb_wait_front(cb_out, 1);
            volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_out));
            const uint32_t v0 = (col0 + i) * 32;
            for (uint32_t j = 0; j < 32; ++j) {
                const uint32_t v = v0 + j;
                if (v >= V) {
                    break;
                }
                const uint32_t key = fkey(t[j < 16 ? j : 240 + j]);
#ifdef SAMPLE
                if (cnt < K || key > tk[K - 1]) {  // (ascending ids: an equal key stays behind the earlier id)
                    uint32_t pos = cnt < K ? cnt++ : K - 1;
                    while (pos > 0 && tk[pos - 1] < key) {
                        tk[pos] = tk[pos - 1];
                        ti[pos] = ti[pos - 1];
                        --pos;
                    }
                    tk[pos] = key;
                    ti[pos] = v;
                }
#else
                if (key > best_key) {
                    best_key = key;
                    best_id = v;
                }
#endif
            }
            cb_pop_front(cb_out, 1);
        }
    }
#ifdef SAMPLE
    // top K -> the reducer (slot * 8 K bytes)
    volatile tt_l1_ptr uint32_t* mine = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cand + ncores * 8 * K);
    for (uint32_t j = 0; j < K; ++j) {
        mine[2 * j] = j < cnt ? tk[j] : 0;
        mine[2 * j + 1] = j < cnt ? ti[j] : 0xFFFFFFFFu;
    }
    noc_async_write(cand + ncores * 8 * K, get_noc_addr(red_x, red_y, cand + slot * 8 * K), 8 * K);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(red_x, red_y, sem), 1);
    noc_async_atomic_barrier();
    if (slot != 0) {
        return;
    }
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem), ncores);
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem), 0);
    volatile tt_l1_ptr uint32_t* c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cand);
    // merge: the lists' heads in local arrays (L1 reads only on a pop)
    uint32_t hk[ncores], hid[ncores];
    uint8_t head[ncores];
    for (uint32_t q = 0; q < ncores; ++q) {
        head[q] = 0;
        hk[q] = c[q * K * 2];
        hid[q] = c[q * K * 2 + 1];
    }
#ifndef LOCAL_OUT
    // out rows: ids at mine[0 .. K), logits at mine[K .. 2K), the T * Gumbel row at mine[2K .. 3K)
    const auto s_g = TensorAccessor(g_args, get_arg_val<uint32_t>(8), 4 * K);
    noc_async_read(s_g.get_noc_addr(KS), cand + ncores * 8 * K + 8 * K, 4 * K);
    const uint32_t tab = cand + ncores * 8 * K + 12 * K;
    noc_async_read(TensorAccessor(t_args, get_arg_val<uint32_t>(11), 4 * NTAB).get_noc_addr(0), tab, 4 * NTAB);
#endif
    for (uint32_t j = 0; j < K; ++j) {
        uint32_t bq = 0, bk = hk[0], bi = hid[0];
        for (uint32_t q = 1; q < ncores; ++q) {
            const uint32_t k = hk[q];
            if (k > bk || (k == bk && hid[q] < bi)) {
                bq = q;
                bk = k;
                bi = hid[q];
            }
        }
        const uint32_t h = ++head[bq];
        hk[bq] = h < K ? c[(bq * K + h) * 2] : 0;  // (exhausted: key 0, id ~0 never wins over a live entry)
        hid[bq] = h < K ? c[(bq * K + h) * 2 + 1] : 0xFFFFFFFFu;
#ifdef LOCAL_OUT
        tk[j] = bk;
#else
        tk[j] = (bk & 0x80000000u) ? (bk & 0x7FFFFFFFu) : ~bk;  // the FP32 bits back
#endif
        ti[j] = bi;
    }
#ifdef LOCAL_OUT
    write_loc(TensorAccessor(w_args, get_arg_val<uint32_t>(6), 4096), tk, ti, K, cand + W1_OFF);
#else
    noc_async_read_barrier();
    volatile tt_l1_ptr float* tg = reinterpret_cast<volatile tt_l1_ptr float*>(cand + ncores * 8 * K + 8 * K);
    volatile tt_l1_ptr float* ct = reinterpret_cast<volatile tt_l1_ptr float*>(tab);
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
        mine[j] = ti[j];
        mine[K + j] = cv.u;
        const float v = cv.f + tg[j];
        if (j == 0 || v > bv) {
            bv = v;
            best = j;
        }
    }
    noc_async_write(cand + ncores * 8 * K, TensorAccessor(ci_args, get_arg_val<uint32_t>(9), 4 * K).get_noc_addr(KS),
                    4 * K);
    noc_async_write(cand + ncores * 8 * K + 4 * K,
                    TensorAccessor(cl_args, get_arg_val<uint32_t>(10), 4 * K).get_noc_addr(KS), 4 * K);
    noc_async_write_barrier();
    mine[0] = ti[best];
    const auto s_p = TensorAccessor(p_args, prev_addr, 4);
    noc_async_write(cand + ncores * 8 * K, s_p.get_noc_addr(0), 4);
    noc_async_write_barrier();
    w1_row(TensorAccessor(w_args, get_arg_val<uint32_t>(6), 512), TensorAccessor(wp_args, get_arg_val<uint32_t>(7), 2048),
           ti[best], cand + W1_OFF);
#endif
#else
    // candidate -> the reducer
    volatile tt_l1_ptr uint32_t* mine = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cand + ncores * 16);
    mine[0] = best_key;
    mine[1] = best_id;
    noc_async_write(cand + ncores * 16, get_noc_addr(red_x, red_y, cand + slot * 16), 16);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(red_x, red_y, sem), 1);
    noc_async_atomic_barrier();
    if (slot != 0) {
        return;
    }
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem), ncores);
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem), 0);
    volatile tt_l1_ptr uint32_t* c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cand);
    uint32_t bk = 0, bi = 0xFFFFFFFFu;
    for (uint32_t s = 0; s < ncores; ++s) {
        const uint32_t k = c[s * 4], id = c[s * 4 + 1];
        if (k > bk || (k == bk && id < bi)) {
            bk = k;
            bi = id;
        }
    }
#ifdef LOCAL_OUT
    write_loc(TensorAccessor(w_args, get_arg_val<uint32_t>(6), 4096), &bk, &bi, 1, cand + W1_OFF);
#else
    mine[0] = bi;
    const auto s_p = TensorAccessor(p_args, prev_addr, 4);
    noc_async_write(cand + ncores * 16, s_p.get_noc_addr(0), 4);
    noc_async_write_barrier();
    w1_row(TensorAccessor(w_args, get_arg_val<uint32_t>(6), 512), TensorAccessor(wp_args, get_arg_val<uint32_t>(7), 2048),
           bi, cand + W1_OFF);
#endif
#endif
}
