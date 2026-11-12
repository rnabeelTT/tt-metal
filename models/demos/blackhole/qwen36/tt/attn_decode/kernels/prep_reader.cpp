// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Attention decode prep, reader. Work item u = PARTS b + part (part 0: q, 1: k (+ v for the writer), 2: gate; PARTS =
// 2: no gate item, for ungated attention):
//   * the part's heads of row b of the fused [q | k | v | gate] projection ([1, 1, B, W] TILE bf16), head h -> row h of
//     Ht tiles (two 32-byte face-row reads per head and tile), into tiles that are zero elsewhere;
//   * q / k: cos and sin of user b (tiles b * 2 + t of [1, B, 1, 2 * 32] TILE: row 0 is the user's row);
//   * k: v's heads the same way into cb_v (straight to the writer).
// Once per launch: the (1 + w) rows of q and k (row 0 of 2 Ht FP32 tiles) and the all-ones tile.
// Rows other than the heads' are left as they are (never read downstream).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace {
constexpr uint32_t BF_FACE = 512, BF_ROW = 32, BF_TILE = 2048;
constexpr uint32_t F32_FACE = 1024, F32_ROW = 64, F32_TILE = 4096;

inline uint32_t row_off(uint32_t r, uint32_t c, uint32_t face, uint32_t row) {
    return ((r >> 4) * 2 + c) * face + (r & 15) * row;
}

// heads [h0, h0 + n) of row b, starting at qkv tile column c0 (head h at c0 + (h - h0) * Ht), into rows 0..n-1 of the
// Ht tiles at l1. A DRAM read keeps the source address modulo 64 (Blackhole), and the 32-byte rows land at other offsets
// modulo 64: read the 576-byte span holding the row's two face rows (64-byte aligned) into scr, then copy the rows on
// the RISC.
constexpr uint32_t SPAN = BF_FACE + 64, SLOT = SPAN + 64;
template <typename Acc>
inline void gather_heads(
    const Acc& src, uint32_t l1, uint32_t scr, uint32_t wq, uint32_t b, uint32_t c0, uint32_t n, uint32_t Ht) {
    const uint32_t page0 = (b >> 5) * wq;
    const uint32_t off = row_off(b & 31, 0, BF_FACE, BF_ROW);
    const uint32_t al = off & ~63u;
    for (uint32_t h = 0; h < n; ++h) {
        for (uint32_t j = 0; j < Ht; ++j) {
            noc_async_read(src.get_noc_addr(page0 + c0 + h * Ht + j, al), scr + (h * Ht + j) * SLOT, SPAN);
        }
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    for (uint32_t h = 0; h < n; ++h) {
        for (uint32_t j = 0; j < Ht; ++j) {
            for (uint32_t c = 0; c < 2; ++c) {
                const uint32_t* s = reinterpret_cast<const uint32_t*>(scr + (h * Ht + j) * SLOT + off - al + c * BF_FACE);
                uint32_t* d = reinterpret_cast<uint32_t*>(l1 + j * BF_TILE + row_off(h, c, BF_FACE, BF_ROW));
                const uint32_t w0 = s[0], w1 = s[1], w2 = s[2], w3 = s[3], w4 = s[4], w5 = s[5], w6 = s[6], w7 = s[7];
                d[0] = w0, d[1] = w1, d[2] = w2, d[3] = w3, d[4] = w4, d[5] = w5, d[6] = w6, d[7] = w7;
            }
        }
    }
}
}  // namespace

void kernel_main() {
    const uint32_t qkv_addr = get_arg_val<uint32_t>(0);
    const uint32_t cos_addr = get_arg_val<uint32_t>(1);
    const uint32_t sin_addr = get_arg_val<uint32_t>(2);
    const uint32_t w_addr = get_arg_val<uint32_t>(3);
    const uint32_t ones_addr = get_arg_val<uint32_t>(4);
    const uint32_t start = get_arg_val<uint32_t>(5);
    const uint32_t count = get_arg_val<uint32_t>(6);

    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);
    constexpr uint32_t NKV = get_compile_time_arg_val(2);
    constexpr uint32_t wq = get_compile_time_arg_val(3);
    constexpr uint32_t cb_x = get_compile_time_arg_val(4);
    constexpr uint32_t cb_w = get_compile_time_arg_val(5);
    constexpr uint32_t cb_cs = get_compile_time_arg_val(6);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(7);
    constexpr uint32_t cb_v = get_compile_time_arg_val(8);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(9);
    constexpr auto qkv_args = TensorAccessorArgs<10>();
    constexpr auto cos_args = TensorAccessorArgs<qkv_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<sin_args.next_compile_time_args_offset()>();
    constexpr auto ones_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr uint32_t PARTS = get_compile_time_arg_val(ones_args.next_compile_time_args_offset());
    // NORM = 2 (full-width QK norm, MiniMax-M2): q / k items also get the user's partial sums of squares (row b, column
    // 0 of the gathered [1, 1, 32, ND * P * 32] FP32 statistics: tiles d * P + i, i < Pq for q, i = Pq for k) in row 0
    // of a stats tile (cb_st, zero elsewhere in that row); the (gamma) constant is read whole (rows = heads)
    constexpr uint32_t A0 = ones_args.next_compile_time_args_offset() + 1;
    constexpr uint32_t NORM = get_compile_time_arg_val(A0);
    constexpr uint32_t Pq = get_compile_time_arg_val(A0 + 1);
    constexpr uint32_t P = get_compile_time_arg_val(A0 + 2);
    constexpr uint32_t ND = get_compile_time_arg_val(A0 + 3);
    constexpr uint32_t cb_st = get_compile_time_arg_val(A0 + 4);
    constexpr auto st_args = TensorAccessorArgs<A0 + 5>();
    const uint32_t st_addr = get_arg_val<uint32_t>(7);
    if (count == 0) {
        return;
    }
    const auto s_qkv = TensorAccessor(qkv_args, qkv_addr, BF_TILE);
    const auto s_cos = TensorAccessor(cos_args, cos_addr, BF_TILE);
    const auto s_sin = TensorAccessor(sin_args, sin_addr, BF_TILE);
    const auto s_w = TensorAccessor(w_args, w_addr, F32_TILE);
    const auto s_ones = TensorAccessor(ones_args, ones_addr, BF_TILE);

    // constants: issued now, pushed with the first item (its gather waits for every outstanding read)
    cb_reserve_back(cb_w, 2 * Ht);
    const uint32_t w = get_write_ptr(cb_w);
    for (uint32_t t = 0; t < 2 * Ht; ++t) {  // row 0 only (the compute broadcasts row 0); NORM 2: whole tiles
        if constexpr (NORM == 2) {
            noc_async_read(s_w.get_noc_addr(t), w + t * F32_TILE, F32_TILE);
            continue;
        }
        for (uint32_t c = 0; c < 2; ++c) {
            noc_async_read(s_w.get_noc_addr(t, c * F32_FACE), w + t * F32_TILE + c * F32_FACE, F32_ROW);
        }
    }
    const auto s_st = TensorAccessor(st_args, st_addr, F32_TILE);
    cb_reserve_back(cb_ones, 1);
    noc_async_read(s_ones.get_noc_addr(0), get_write_ptr(cb_ones), BF_TILE);

    const uint32_t scr = get_write_ptr(cb_scr);  // scratch, never pushed
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t u = start + i;
        const uint32_t b = u / PARTS;
        const uint32_t part = u - PARTS * b;
        DeviceZoneScopedN("rd_item");
        if (part < 2) {
            cb_reserve_back(cb_cs, 4);
            const uint32_t cs = get_write_ptr(cb_cs);
            for (uint32_t t = 0; t < 2; ++t) {
                noc_async_read(s_cos.get_noc_addr(b * 2 + t), cs + t * BF_TILE, BF_TILE);
                noc_async_read(s_sin.get_noc_addr(b * 2 + t), cs + (2 + t) * BF_TILE, BF_TILE);
            }
        }
        if constexpr (NORM == 2) {
            if (part < 2) {
                // 64-byte aligned reads of (row b, column 0) of each partial tile into scr, then the values (bit
                // copies)
                const uint32_t off = row_off(b & 31, 0, F32_FACE, F32_ROW), al = off & ~63u;
                const uint32_t n = part == 0 ? ND * Pq : ND;
                for (uint32_t i = 0; i < n; ++i) {
                    constexpr uint32_t PQ = Pq ? Pq : 1;
                    const uint32_t t = part == 0 ? (i / PQ) * P + i % PQ : i * P + Pq;
                    noc_async_read(s_st.get_noc_addr(t, al), scr + i * 64, 64);
                }
                noc_async_read_barrier();
                invalidate_l1_cache();
                cb_reserve_back(cb_st, 1);
                volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_st));
                for (uint32_t c = 0; c < 32; ++c) {  // row 0: face 0 columns 0..15, face 1 columns 16..31
                    const uint32_t idx = (c >> 4) * (F32_FACE / 4) + (c & 15);
                    st[idx] = c < n ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + c * 64 + off - al)[0] : 0u;
                }
                cb_push_back(cb_st, 1);
            }
        }
        cb_reserve_back(cb_x, Ht);
        const uint32_t x = get_write_ptr(cb_x);
        if (part == 0) {
            gather_heads(s_qkv, x, scr, wq, b, 0, NH, Ht);
        } else if (part == 1) {
            gather_heads(s_qkv, x, scr, wq, b, NH * Ht, NKV, Ht);
        } else {
            gather_heads(s_qkv, x, scr, wq, b, (NH + 2 * NKV) * Ht, NH, Ht);
        }
        noc_async_read_barrier();
        if (i == 0) {
            cb_push_back(cb_w, 2 * Ht);
            cb_push_back(cb_ones, 1);
        }
        cb_push_back(cb_x, Ht);
        if (part < 2) {
            cb_push_back(cb_cs, 4);
        }
        if (part == 1) {
            cb_reserve_back(cb_v, Ht);
            gather_heads(s_qkv, get_write_ptr(cb_v), scr, wq, b, (NH + NKV) * Ht, NKV, Ht);
            noc_async_read_barrier();
            cb_push_back(cb_v, Ht);
        }
    }
}
