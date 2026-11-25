// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Decode attention prep (tt/aprep.py), writer: unit (b, type) -> its bf16 output tiles: q [1, B, NH, HD] (tiles
// b * HT + c), r [1, B, NH, 32] (b), k / v [1, B, 32, HD] (b * HT + c; the KV-write layout, may be height-sharded),
// sink [1, B, NH, 32] (b).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "aprep_units.hpp"

#ifndef APREP_RB
#define APREP_RB 0
#endif
#ifndef APREP_WB1
#define APREP_WB1 0
#endif

void kernel_main() {
#ifdef APREP_CARGS  // (the addresses in the common runtime args; per core: start, step, total)
#define APREP_ADDR(i) get_common_arg_val<uint32_t>(i)
#define APREP_KVA(i) get_common_arg_val<uint32_t>(5 + (i))
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t step = get_arg_val<uint32_t>(1);
    const uint32_t total = get_arg_val<uint32_t>(2);
#else
#define APREP_ADDR(i) get_arg_val<uint32_t>(i)
#define APREP_KVA(i) get_arg_val<uint32_t>(8 + (i))
    const uint32_t start = get_arg_val<uint32_t>(5);
    const uint32_t step = get_arg_val<uint32_t>(6);
    const uint32_t total = get_arg_val<uint32_t>(7);
#endif
    const uint32_t q_addr = APREP_ADDR(0);
    const uint32_t r_addr = APREP_ADDR(1);
    const uint32_t k_addr = APREP_ADDR(2);
    const uint32_t v_addr = APREP_ADDR(3);
    const uint32_t s_addr = APREP_ADDR(4);
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t HT = get_compile_time_arg_val(1);
    constexpr uint32_t BF = 2048;
    constexpr auto q_args = TensorAccessorArgs<2>();
    constexpr auto r_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto kv_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    const auto aq = TensorAccessor(q_args, q_addr, BF);
    const auto ar = TensorAccessor(r_args, r_addr, BF);
    const auto as = TensorAccessor(r_args, s_addr, BF);
    const auto ak = TensorAccessor(kv_args, k_addr, BF);
    const auto av = TensorAccessor(kv_args, v_addr, BF);
#if APREP_KV
    // k / v rows straight into the paged caches (bf16 [blocks, NKV, BLK, HD]): slot b's write position p (int32 [B]),
    // its block table row (int32 [B, max blocks]); head h's row of tile c -> cache tile ((blk NKV + h) BT + p % BLK / 32)
    // HT + c, row p % 32 (faces 0 / 1 or 2 / 3). DRAM writes keep the source 64 B-aligned like the destination.
    constexpr auto c_args = TensorAccessorArgs<kv_args.next_compile_time_args_offset()>();  // caches
    constexpr auto p_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();  // positions
    constexpr auto t_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();  // block table
    const auto ack = TensorAccessor(c_args, APREP_KVA(0), BF);
    const auto acv = TensorAccessor(c_args, APREP_KVA(1), BF);
    const auto apos = TensorAccessor(p_args, APREP_KVA(2), APREP_POS_PAGE);
    const auto atab = TensorAccessor(t_args, APREP_KVA(3), APREP_TAB_PAGE);
    constexpr uint32_t cb_st = APREP_CB_STAGE;
    cb_reserve_back(cb_st, 1);
    const uint32_t st = get_write_ptr(cb_st);  // [0, 256): positions page; [256, 320): table chunk; [512, ...): rows

#endif
    for (uint32_t u = start; u < total; u += step) {
        const uint32_t b = u / APREP_NU;
        uint32_t type, h0, h1;
        aprep_unit(u % APREP_NU, APREP_NH, APREP_NKVH, type, h0, h1);
        const uint32_t nt = (type == 1 || type == 4) ? 1 : HT;
#if APREP_KV
        if (type == 2 || type == 3) {
#if APREP_FILL
            // prefill: row b of each head into the L1 [1, NKV, T, HD] fill tensors (ack / acv: tile (h T / 32 + b / 32)
            // HT + c, row b % 32); one paged_fill_cache per cache follows. (Not straight into the DRAM caches: rows
            // 2k / 2k + 1 share a 64-byte DRAM word and come from different cores.)
            const uint32_t r = b & 31;
            const auto& ac = type == 2 ? ack : acv;
            for (uint32_t c = 0; c < HT; ++c) {
                cb_wait_front(cb_out, 1);
                const uint32_t l1 = get_read_ptr(cb_out);
                for (uint32_t h = 0; h < APREP_NKV; ++h) {
                    const uint64_t tile = ac.get_noc_addr((h * APREP_RT + (b >> 5)) * HT + c);
                    for (uint32_t f = 0; f < 2; ++f) {
                        noc_async_write(l1 + ((h >> 4) * 2 + f) * 512 + (h & 15) * 32,
                                        tile + ((r >> 4) * 2 + f) * 512 + (r & 15) * 32, 32);
                    }
                }
                noc_async_write_barrier();
                cb_pop_front(cb_out, 1);
            }
            continue;
#else
            noc_async_read(apos.get_noc_addr(0), st, APREP_POS_PAGE);
            noc_async_read_barrier();
            const uint32_t p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(st)[b];
            const uint32_t to = (p / APREP_BLK) * 4;  // the entry's byte offset in row b of the table
            noc_async_read(atab.get_noc_addr(b) + (to & ~63u), st + 256, 64);
            noc_async_read_barrier();
            const uint32_t blk = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(st + 256 + (to & 63u))[0];
            const uint32_t row = p % APREP_BLK, r = row % 32;
            const uint32_t face = r < 16 ? r * 32 : 1024 + (r - 16) * 32;
            const auto& ac = type == 2 ? ack : acv;
            const uint32_t RB0 = APREP_RB ? h0 : 0;
            // every tile's head rows staged at once (one read barrier, one write barrier)
            cb_wait_front(cb_out, HT);
            const uint32_t l1 = get_read_ptr(cb_out);
            for (uint32_t c = 0; c < HT; ++c) {
                for (uint32_t h = h0; h < h1; ++h) {
                    for (uint32_t f = 0; f < 2; ++f) {  // (left / right face of the row)
                        const uint32_t sl = st + 512 + ((c * APREP_NKV + h) * 2 + f) * 64 + (r & 1) * 32;
                        noc_async_read(get_noc_addr(l1 + c * BF + f * 512 + (h - RB0) * 32), sl, 32);
                    }
                }
            }
            noc_async_read_barrier();
            for (uint32_t c = 0; c < HT; ++c) {
                for (uint32_t h = h0; h < h1; ++h) {
                    const uint64_t tile =
                        ac.get_noc_addr(((blk * APREP_NKV + h) * (APREP_BLK / 32) + row / 32) * HT + c);
                    for (uint32_t f = 0; f < 2; ++f) {
                        noc_async_write(st + 512 + ((c * APREP_NKV + h) * 2 + f) * 64 + (r & 1) * 32,
                                        tile + face + f * 512, 32);
                    }
                }
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, HT);
            continue;
#endif
        }
#endif
        if ((type == 0 && APREP_QN > 1) || ((type == 2 || type == 3) && APREP_KN > 1)) {
            // rows [h0, h1) of each tile (a head group: inside one 16-row face pair)
            const uint32_t n = h1 - h0, o = (h0 >> 4) * 1024 + (h0 & 15) * 32, so = APREP_RB ? 0 : o;
#if APREP_WB1  // (all of the unit's tiles, one barrier: the compute pushes them together)
            cb_wait_front(cb_out, nt);
            for (uint32_t c = 0; c < nt; ++c) {
                const uint32_t l1 = get_read_ptr(cb_out) + c * BF;
#else
            for (uint32_t c = 0; c < nt; ++c) {
                cb_wait_front(cb_out, 1);
                const uint32_t l1 = get_read_ptr(cb_out);
#endif
                const uint64_t dst = type == 0 ? aq.get_noc_addr(b * HT + c)
                                     : type == 2 ? ak.get_noc_addr(b * HT + c) : av.get_noc_addr(b * HT + c);
                noc_async_write(l1 + so, dst + o, n * 32);
                noc_async_write(l1 + so + 512, dst + o + 512, n * 32);
#if !APREP_WB1
                noc_async_write_barrier();
                cb_pop_front(cb_out, 1);
#endif
            }
#if APREP_WB1
            noc_async_write_barrier();
            cb_pop_front(cb_out, nt);
#endif
            continue;
        }
        for (uint32_t c = 0; c < nt; ++c) {
            cb_wait_front(cb_out, 1);
            const uint32_t l1 = get_read_ptr(cb_out);
            uint64_t dst;
            if (type == 0) {
                dst = aq.get_noc_addr(b * HT + c);
            } else if (type == 1) {
                dst = ar.get_noc_addr(b);
            } else if (type == 2) {
                dst = ak.get_noc_addr(b * HT + c);
            } else if (type == 3) {
                dst = av.get_noc_addr(b * HT + c);
            } else {
                dst = as.get_noc_addr(b);
            }
            noc_async_write(l1, dst, BF);
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
