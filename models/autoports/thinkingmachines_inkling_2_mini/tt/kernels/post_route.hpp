// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The MoE routing inside a router-round program (post / gate2, POST_ROUTE / G2_ROUTE with the router round split:
// root (t, e) holds logit tile e of row t, row 0 of a TH x 32 fp32 CB tile). qwen36 RouteRows' sigb router
// (router_topk.hpp, ROUTE_KEYS + ROUTE_FAST): selection keys sigmoid_q23(logit) + bias + 2^28, top-k sorted, ties to the
// lower id, weights s_j / sum s from the keys. Root (t, e): its BRISC the experts 32 e .. + 15, its NCRISC the next
// 16 (route_part), merged; root (t, e > 0) sends its list to root (t, 0), which merges the ET lists in expert order
// and writes row t's 64-byte route record (k ids, shared weight 1 at 16, k weights at 17..) into the route table (+ row 1:
// the duplicated one-row decode). Every merge keeps the global (key desc, id asc) order: the route program's bits.
// Needs RT_E (experts) and ROUTE_FAST before router_topk.hpp.

#pragma once

#include <stdint.h>

// the route scratch CB: [selection bias Q23 (E) int32][NCRISC's list][slots 0 .. ET - 1][record]
constexpr uint32_t RS_C = 0;
constexpr uint32_t RS_CBYTES = 4 * RT_E;
constexpr uint32_t RS_RL = (RS_CBYTES + 63) & ~63u;
constexpr uint32_t RS_SL = RS_RL + 64;
constexpr uint32_t rs_bytes(uint32_t ET) { return RS_SL + 64 * ET + 64; }

// one RISC's half of root (e0 / 32)'s top-k: part 0 the logits 0..15 (at t3), part 1 16..31 (at t3 + fo); c = bias
// (A pruning pass with per-bucket sigmoid bounds measured no faster: ~2 us for its 32 bounds alone, L1-bound.)
template <uint32_t K>
inline void route_part(uint32_t t3, uint32_t fo, const int32_t* c, uint32_t e0, uint32_t part, uint32_t* keys,
                       uint32_t* ids) {
    const uint32_t a = e0 + 16 * part;
    moe_topk::select_range_sigb<K>(reinterpret_cast<const uint32_t*>(t3 + part * fo) - a, c, a, a + 16, keys, ids);
}

// NCRISC of root (t, e): its half once the BRISC posts the tile's address in s_ta (the bias
// is in by then); its list -> RS_RL, then s_rd = 1
template <uint32_t K>
inline void route_reader(uint32_t fo, uint32_t rs, uint32_t re, volatile tt_l1_ptr uint32_t* s_ta,
                         volatile tt_l1_ptr uint32_t* s_rd) {
    uint32_t t3;
    while ((t3 = *s_ta) == 0) {
        invalidate_l1_cache();
    }
    invalidate_l1_cache();
    GLZ("R_RT");
    uint32_t keys[K], ids[K];
    route_part<K>(t3, fo, reinterpret_cast<const int32_t*>(rs + RS_C), 32 * re, 1, keys, ids);
    volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rs + RS_RL);
    for (uint32_t j = 0; j < K; ++j) {
        o[j] = keys[j];
        o[K + j] = ids[j];
    }
    asm volatile("fence" ::: "memory");
    *s_rd = 1;
    GLZ("R_RS");
}

// BRISC of root (t, e): t3 = the logit tile (row 0), fo its face 1, leader = root (t, 0)'s NOC address (offset 0),
// pop() frees t3; the bias was read into RS_C earlier. The caller resets s_ta / s_rd / s_rg
// (semaphores) at its end
template <uint32_t K, uint32_t ET, uint32_t DUP, typename TA, typename POP>
inline void route_root(uint32_t t3, uint32_t fo, uint32_t rs, uint32_t re, uint32_t row,
                       volatile tt_l1_ptr uint32_t* s_ta, volatile tt_l1_ptr uint32_t* s_rd,
                       volatile tt_l1_ptr uint32_t* s_rg, uint64_t leader, const TA& tab, POP pop) {
    noc_async_read_barrier();  // (the bias)
    invalidate_l1_cache();
    asm volatile("fence" ::: "memory");
    *s_ta = t3;  // (the NCRISC's part starts)
    GLZ("W_RT");
    const int32_t* c = reinterpret_cast<const int32_t*>(rs + RS_C);
    uint32_t keys[K], ids[K];
    route_part<K>(t3, fo, c, 32 * re, 0, keys, ids);
    GLZ("W_RS");
    while (*s_rd == 0) {
        invalidate_l1_cache();
    }
    invalidate_l1_cache();
    const uint32_t* rl = reinterpret_cast<const uint32_t*>(rs + RS_RL);
    moe_topk::merge_into<K>(keys, ids, rl, rl + K);
    pop();
    const uint32_t slot = rs + RS_SL + 64 * re;
    if (re != 0) {
        volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot);
        for (uint32_t j = 0; j < K; ++j) {
            o[j] = keys[j];
            o[K + j] = ids[j];
        }
        asm volatile("fence" ::: "memory");
        noc_async_write(slot, leader + slot, 8 * K);
        noc_async_writes_flushed();
        noc_semaphore_inc(leader + reinterpret_cast<uint32_t>(s_rg), 1);
        return;
    }
    if constexpr (ET > 1) {
        noc_semaphore_wait(s_rg, ET - 1);
        invalidate_l1_cache();
        GLZ("W_RG");
        moe_topk::merge_parts<K, 1, ET>(keys, ids, rs + RS_SL);
    }
    uint16_t w[K];
    moe_topk::sigmoid_weights_keys_fast<K>(keys, ids, c, w);
    const uint32_t rec = rs + RS_SL + 64 * ET;
    volatile tt_l1_ptr uint16_t* r = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rec);
    for (uint32_t j = 0; j < 32; ++j) {
        r[j] = 0;
    }
    for (uint32_t j = 0; j < K; ++j) {
        r[j] = static_cast<uint16_t>(ids[j]);
        r[17 + j] = w[j];
    }
    r[16] = 0x3F80;
    asm volatile("fence" ::: "memory");
    noc_async_write(rec, tab.get_noc_addr(row), 64);
    if constexpr (DUP) {
        noc_async_write(rec, tab.get_noc_addr(1), 64);
    }
    noc_async_write_barrier();
    GLZ("W_RE");
}
