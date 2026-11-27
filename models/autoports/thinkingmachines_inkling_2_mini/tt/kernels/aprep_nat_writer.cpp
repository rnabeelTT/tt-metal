// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Prefill attention prep in the projections' layout (tt/aprep.py NAT), writer: the compute's bf16 tiles (rows = the
// tile row's tokens) -> the SDPA's per-row layouts. A q (r) head group [h0, h1): each head's tiles as they come are
// staged per token (token b, tile c, face half: the group's rows side by side), then token b's rows h0 .. h1 - 1 of
// q tile (32 R + b) HT + c (r tile 32 R + b) go out as one run per face half; a k / v head: its HT tiles whole into
// the L1 fill tensors [1, NKV, T, HD] (tile (h RT + R) HT + c); the sink: token b's sink tile, row h = s_h / scale in
// every column (h < NH). Rows past NH of the q / r / sink tiles are not written (the SDPA reads only its heads' rows).
// NAT_STK: q / rq / sink out in the SDPA's stacked tiles instead (tt/attn/inkling_sdpa.py stk; 8-row items, item i =
// tokens 8 i .. 8 i + 7, token r of the item): q tile ((i NKV + kv head) HT + c) rows r G + g (head kv G + g), rq tile
// ((i NKV + kv) 4 + r / 2) the same rows, columns 0-15 (even r) / 16-31 (odd r), sink tile (i NKV + kv) the same rows,
// every other row zero. The q / r groups are the kv heads, so a unit owns whole tiles: they are assembled in L1 (cb_st:
// 4 HT q, 16 rq, 4 NKV sink tiles, zeroed once; a unit rewrites the same rows each time) and written whole.
// cb_out gets HT-tile pushes only (q / k / v heads), cb_out1 1-tile pushes (r heads, the sink): no wrapped waits.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "aprep_nat.hpp"

void kernel_main() {
    const uint32_t q_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t r_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t s_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t kf_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t vf_addr = get_common_arg_val<uint32_t>(4);
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t step = get_arg_val<uint32_t>(1);
    const uint32_t total = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_st = get_compile_time_arg_val(1);
    constexpr uint32_t HT = get_compile_time_arg_val(2);
    constexpr uint32_t RT = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out1 = get_compile_time_arg_val(4);
    constexpr uint32_t BF = 2048;
    constexpr auto q_args = TensorAccessorArgs<5>();
    constexpr auto r_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto f_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    const auto aq = TensorAccessor(q_args, q_addr, BF);
    const auto ar = TensorAccessor(r_args, r_addr, BF);
    const auto as = TensorAccessor(r_args, s_addr, BF);
    const auto akf = TensorAccessor(f_args, kf_addr, BF);
    const auto avf = TensorAccessor(f_args, vf_addr, BF);

    // staging: per token TOK bytes (HT tiles x 2 face halves x 8 rows x 32 B; or the sink's tile-shaped rows)
    cb_reserve_back(cb_st, 1);
    const uint32_t st0 = get_write_ptr(cb_st);
    constexpr uint32_t TOK = (HT * 2 * 8 * 32 > BF) ? HT * 2 * 8 * 32 : BF;

#if NAT_STK
    constexpr uint32_t QS = 0, RS = 4 * HT, SS = 4 * HT + 16;  // (cb_st tiles: q [m][c], rq [m][p], sink [m][kv])
    {
        const uint64_t z = get_noc_addr(MEM_ZEROS_BASE);
        constexpr uint32_t bytes = (SS + 4 * NAT_NKV) * BF;
        for (uint32_t o = 0; o < bytes; o += MEM_ZEROS_SIZE) {
            noc_async_read(z, st0 + o, bytes - o < MEM_ZEROS_SIZE ? bytes - o : MEM_ZEROS_SIZE);
        }
        noc_async_read_barrier();
    }
    auto cp32 = [](uint32_t from, uint32_t to) {
        const uint32_t* s = reinterpret_cast<const uint32_t*>(from);
        uint32_t* d = reinterpret_cast<uint32_t*>(to);
        const uint32_t w0 = s[0], w1 = s[1], w2 = s[2], w3 = s[3], w4 = s[4], w5 = s[5], w6 = s[6], w7 = s[7];
        d[0] = w0, d[1] = w1, d[2] = w2, d[3] = w3, d[4] = w4, d[5] = w5, d[6] = w6, d[7] = w7;
    };
#endif
    for (uint32_t u = start; u < total; u += step) {
        const uint32_t R = u / NAT_NU;
        uint32_t type, h0, h1;
        nat_unit(u % NAT_NU, type, h0, h1);
#if NAT_STK
        if (type == 0 || type == 1) {
            const uint32_t kv = h0 / NAT_G, nt = type == 0 ? HT : 1, cbi = type == 0 ? cb_out : cb_out1;
            for (uint32_t j = 0; j < NAT_G; ++j) {
                cb_wait_front(cbi, nt);
                invalidate_l1_cache();  // (plain loads below: the packer wrote these tiles)
                const uint32_t l1 = get_read_ptr(cbi);
                for (uint32_t b = 0; b < 32; ++b) {
                    const uint32_t m = b / 8, r = b % 8, row = r * NAT_G + j;
                    if (type == 0) {
                        for (uint32_t c = 0; c < HT; ++c) {
                            for (uint32_t half = 0; half < 2; ++half) {
                                cp32(l1 + c * BF + nat_bf_row(b, half), st0 + (QS + m * HT + c) * BF + nat_bf_row(row, half));
                            }
                        }
                    } else {  // (r: columns 0-15 -> the row's column half)
                        cp32(l1 + nat_bf_row(b, 0), st0 + (RS + m * 4 + r / 2) * BF + nat_bf_row(row, r & 1));
                    }
                }
                cb_pop_front(cbi, nt);
            }
            const auto& a = type == 0 ? aq : ar;
            const uint32_t per = type == 0 ? HT : 4, t0 = type == 0 ? QS : RS;
            for (uint32_t m = 0; m < 4; ++m) {
                for (uint32_t c = 0; c < per; ++c) {
                    noc_async_write(st0 + (t0 + m * per + c) * BF, a.get_noc_addr(((R * 4 + m) * NAT_NKV + kv) * per + c), BF);
                }
            }
            noc_async_write_barrier();
            continue;
        }
        if (type == 4) {
            cb_wait_front(cb_out1, 1);
            invalidate_l1_cache();
            const uint16_t* src = reinterpret_cast<const uint16_t*>(get_read_ptr(cb_out1));
            for (uint32_t b = 0; b < 32; ++b) {
                const uint32_t m = b / 8, r = b % 8;
                for (uint32_t h = 0; h < NAT_NH; ++h) {
                    const uint32_t v = src[((b >> 4) * 2 + (h >> 4)) * 256 + (b & 15) * 16 + (h & 15)];
                    const uint32_t vv = v | (v << 16);
                    const uint32_t kv = h / NAT_G, row = r * NAT_G + h % NAT_G;
                    for (uint32_t half = 0; half < 2; ++half) {
                        uint32_t* dw = reinterpret_cast<uint32_t*>(st0 + (SS + m * NAT_NKV + kv) * BF + nat_bf_row(row, half));
                        for (uint32_t w = 0; w < 8; ++w) {
                            dw[w] = vv;
                        }
                    }
                }
            }
            for (uint32_t m = 0; m < 4; ++m) {
                for (uint32_t kv = 0; kv < NAT_NKV; ++kv) {
                    noc_async_write(st0 + (SS + m * NAT_NKV + kv) * BF, as.get_noc_addr((R * 4 + m) * NAT_NKV + kv), BF);
                }
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out1, 1);
            continue;
        }
#endif
        if (type == 2 || type == 3) {
            const auto& a = type == 2 ? akf : avf;
            cb_wait_front(cb_out, HT);
            const uint32_t l1 = get_read_ptr(cb_out);
            for (uint32_t c = 0; c < HT; ++c) {
                noc_async_write(l1 + c * BF, a.get_noc_addr((h0 * RT + R) * HT + c), BF);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, HT);
            continue;
        }
        if (type == 4) {
            cb_wait_front(cb_out1, 1);
            invalidate_l1_cache();
            const uint16_t* src = reinterpret_cast<const uint16_t*>(get_read_ptr(cb_out1));
            for (uint32_t b = 0; b < 32; ++b) {
                const uint32_t st = st0 + b * TOK;
                uint32_t* dw = reinterpret_cast<uint32_t*>(st);
                for (uint32_t h = 0; h < NAT_NH; ++h) {
                    // element (b, h) of the natural tile
                    const uint32_t v = src[((b >> 4) * 2 + (h >> 4)) * 256 + (b & 15) * 16 + (h & 15)];
                    const uint32_t vv = v | (v << 16);
                    for (uint32_t half = 0; half < 2; ++half) {
                        const uint32_t o = nat_bf_row(h, half) / 4;
                        for (uint32_t w = 0; w < 8; ++w) {
                            dw[o + w] = vv;
                        }
                    }
                }
                const uint64_t dst = as.get_noc_addr(R * 32 + b);
                for (uint32_t top = 0; top < NAT_NH; top += 16) {
                    const uint32_t n = NAT_NH - top < 16 ? NAT_NH - top : 16;
                    for (uint32_t half = 0; half < 2; ++half) {
                        const uint32_t o = nat_bf_row(top, half);
                        noc_async_write(st + o, dst + o, n * 32);
                    }
                }
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out1, 1);
            continue;
        }
        // q / r head group: head j's nt tiles -> staging (token b, tile c, half: row j)
        const uint32_t nt = type == 0 ? HT : 1;
        const uint32_t cbi = type == 0 ? cb_out : cb_out1;
        const uint32_t n = h1 - h0;
        for (uint32_t j = 0; j < n; ++j) {
            cb_wait_front(cbi, nt);
            invalidate_l1_cache();  // (plain loads below: the packer wrote these tiles; the cache is write-through)
            const uint32_t l1 = get_read_ptr(cbi);
            for (uint32_t c = 0; c < nt; ++c) {
                for (uint32_t b = 0; b < 32; ++b) {
                    for (uint32_t half = 0; half < 2; ++half) {
                        const uint32_t* s = reinterpret_cast<const uint32_t*>(l1 + c * BF + nat_bf_row(b, half));
                        uint32_t* d = reinterpret_cast<uint32_t*>(st0 + b * TOK + (c * 2 + half) * 256 + j * 32);
                        const uint32_t w0 = s[0], w1 = s[1], w2 = s[2], w3 = s[3], w4 = s[4], w5 = s[5], w6 = s[6],
                                       w7 = s[7];
                        d[0] = w0, d[1] = w1, d[2] = w2, d[3] = w3, d[4] = w4, d[5] = w5, d[6] = w6, d[7] = w7;
                    }
                }
            }
            cb_pop_front(cbi, nt);
        }
        const auto& a = type == 0 ? aq : ar;
        for (uint32_t b = 0; b < 32; ++b) {
            for (uint32_t c = 0; c < nt; ++c) {
                const uint64_t dst = a.get_noc_addr((R * 32 + b) * nt + c);
                for (uint32_t half = 0; half < 2; ++half) {
                    noc_async_write(st0 + b * TOK + (c * 2 + half) * 256, dst + nat_bf_row(h0, half), n * 32);
                }
            }
        }
        noc_async_write_barrier();
    }
    noc_async_write_barrier();
}
