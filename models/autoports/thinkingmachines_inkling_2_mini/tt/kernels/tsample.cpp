// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Decode sampling on device (tt/sampling.DeviceSampler): one group of <= 8 rows of the top-K candidates (tt/sampling.py
// DeviceTopK: indices uint32 TILE, logits - row max bf16 TILE, NT tiles per row: one per die of a vocab-sharded head)
// -> each row's token = argmax_j (logit_j + tg[row, j]) over the candidates with id < V, first j on ties, into
// tokens[G0 + row] (uint32 [1, W] ROW_MAJOR: the next step's token input). Indices uint32 or (a slice of <= 65536
// columns) uint16: IB bytes. tg = T * Gumbel noise written by the host
// (zeros: greedy). NT > 1: the dies' candidates merged first like DeviceTopK.host (top K by logit, ties: lowest id), on
// integer keys (no soft-float compares): one row 233 -> ~10 us on 4 dies, 107 -> ~5 on 2; tokens bitwise.
//
// runtime args: idx_addr, val_addr, tg_addr, tok_addr

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t ROWS = get_compile_time_arg_val(1);
    constexpr uint32_t G0 = get_compile_time_arg_val(2);
    constexpr uint32_t W = get_compile_time_arg_val(3);
    constexpr uint32_t NT = get_compile_time_arg_val(4);
    constexpr uint32_t VP = get_compile_time_arg_val(5);
    constexpr uint32_t V = get_compile_time_arg_val(6);
    constexpr uint32_t IB = get_compile_time_arg_val(7);
    constexpr uint32_t K = 32;
    constexpr auto i_args = TensorAccessorArgs<8>();
    constexpr auto v_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto t_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();

    const uint32_t l1 = get_write_ptr(cb);
    const uint32_t li = l1, lv = li + NT * 1024 * IB, lg = lv + NT * 2048, lt = lg + ROWS * 128, lc = lt + 128;
    const auto s_i = TensorAccessor(i_args, get_arg_val<uint32_t>(0), 1024 * IB);
    const auto s_v = TensorAccessor(v_args, get_arg_val<uint32_t>(1), 2048);
    const auto s_g = TensorAccessor(g_args, get_arg_val<uint32_t>(2), 128);
    const auto s_t = TensorAccessor(t_args, get_arg_val<uint32_t>(3), 4 * W);
    for (uint32_t t = 0; t < NT; ++t) {
        noc_async_read(s_i.get_noc_addr(t), li + t * 1024 * IB, 1024 * IB);
        noc_async_read(s_v.get_noc_addr(t), lv + t * 2048, 2048);
    }
    for (uint32_t r = 0; r < ROWS; ++r) {
        noc_async_read(s_g.get_noc_addr(G0 + r), lg + r * 128, 128);
    }
    noc_async_read(s_t.get_noc_addr(0), lt, 4 * W);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(li);
    volatile tt_l1_ptr uint16_t* ids16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(li);
    volatile tt_l1_ptr uint16_t* vals = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(lv);
    volatile tt_l1_ptr float* tg = reinterpret_cast<volatile tt_l1_ptr float*>(lg);
    volatile tt_l1_ptr uint32_t* tok = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lt);
    union {
        uint32_t u;
        float f;
    } cv;
    for (uint32_t r = 0; r < ROWS; ++r) {
        // (the merged candidates in L1 scratch, not on the small kernel stack). The merge compares integer keys: a bf16
        // value's bits mapped so unsigned order = float order (-0 as +0)
        uint32_t* cid = reinterpret_cast<uint32_t*>(lc);
        uint32_t* ckey = reinterpret_cast<uint32_t*>(lc + NT * K * 4);
        for (uint32_t t = 0; t < NT; ++t) {
            for (uint32_t c = 0; c < K; ++c) {
                const uint32_t e = ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
                cid[t * K + c] = (IB == 4 ? ids[t * 1024 + e] : static_cast<uint32_t>(ids16[t * 1024 + e])) + t * VP;
                uint32_t u = static_cast<uint32_t>(vals[t * 2048 / 2 + e]) << 16;
                u = (u << 1) == 0 ? 0 : u;
                ckey[t * K + c] = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
            }
        }
        uint32_t oid[K];
        float olg[K];
        // top K over the dies (logit descending, ties: lowest id): each die's K sorted in that order (insertion sort:
        // the top-k output is nearly sorted), then an NT-way merge
        if constexpr (NT > 1) {
            for (uint32_t t = 0; t < NT; ++t) {
                uint32_t* ks = ckey + t * K;
                uint32_t* is = cid + t * K;
                for (uint32_t c = 1; c < K; ++c) {
                    const uint32_t kc = ks[c], ic = is[c];
                    uint32_t d = c;
                    while (d > 0 && (ks[d - 1] < kc || (ks[d - 1] == kc && is[d - 1] > ic))) {
                        ks[d] = ks[d - 1];
                        is[d] = is[d - 1];
                        --d;
                    }
                    ks[d] = kc;
                    is[d] = ic;
                }
            }
        }
        uint32_t head[NT];
        for (uint32_t t = 0; t < NT; ++t) {
            head[t] = t * K;
        }
        for (uint32_t j = 0; j < K; ++j) {
            uint32_t b = 0xFFFFFFFFu;
            for (uint32_t t = 0; t < NT; ++t) {
                const uint32_t q = head[t];
                if (q == (t + 1) * K) {  // (this die's K all taken)
                    continue;
                }
                if (b == 0xFFFFFFFFu || ckey[q] > ckey[b] || (ckey[q] == ckey[b] && cid[q] < cid[b])) {
                    b = q;
                }
            }
            ++head[b / K];
            oid[j] = cid[b];
            cv.u = (ckey[b] & 0x80000000u) ? (ckey[b] & 0x7FFFFFFFu) : ~ckey[b];
            olg[j] = cv.f;
        }
        uint32_t best = 0xFFFFFFFFu;
        float bv = 0.0f;
        for (uint32_t j = 0; j < K; ++j) {
            if (oid[j] >= V) {  // (the LM head's zero-padded vocab columns)
                continue;
            }
            const float s = olg[j] + tg[r * K + j];
            if (best == 0xFFFFFFFFu || s > bv) {
                bv = s;
                best = j;
            }
        }
        tok[G0 + r] = best == 0xFFFFFFFFu ? 0 : oid[best];
    }
    noc_async_write(lt, s_t.get_noc_addr(0), 4 * W);
    noc_async_write_barrier();
}
