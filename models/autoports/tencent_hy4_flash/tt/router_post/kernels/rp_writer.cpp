// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Router post, writer: key tile t -> output page t, s tile t -> page n + t (the [keys | s] row, one tile row).
// runtime args: [y, t0, t1, (RP_ROUTE) table, core index c, group g]; compile time after the table's accessor args:
// (NoC x | y << 8) of core g NC + q
//
// RP_ROUTE (tt/router_post route=, one key tile per core, RP_GROUPS copies of the NC cores): the routing kernels' top-k
// + weights here instead of in the MoE's programs (qwen36 moe_decode router_topk.hpp, "keys" router). Group g takes the
// real rows r = g mod RP_GROUPS (local row l = r / RP_GROUPS); per row: this core's top-k of its 32 keys (order keys of
// the FP32 bits, ties to the lower id: select_range_keys' total order), with their s values, to the row owner's list
// slot c (owner: the group's core l % NC); the owner takes the top-k of the NC lists scanned in core order (ascending
// id ranges: ties still to the lower id; = the gate|up tail's two halves merged), the weights s_j / sum s as
// keys_weights, and writes the row's route record (ids [0, k), shared weight 16 (1.0), weights [17, 17 + k),
// RP_NLIVE: the live count at R_NLIVE as the one-row gate|up op writes it) to row r of the table (REC_BYTES pages).
// RP_WEXP: the weights' BF16 exponent + RP_WEXP (x 2^RP_WEXP, exact: the routed scale folded in when it is a power of
// two in BF16, as ttnn.multiply's BF16 scalar).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#ifdef RP_ROUTE
#include "../../../../../demos/blackhole/qwen36/tt/moe_decode/kernels/router_topk.hpp"
#endif

void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t t0 = get_arg_val<uint32_t>(1);
    const uint32_t t1 = get_arg_val<uint32_t>(2);
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr auto y_args = TensorAccessorArgs<1>();
    constexpr uint32_t F32 = 4096, cb_k = 16, cb_s = 17;
    const auto s_y = TensorAccessor(y_args, y_addr, F32);
#ifndef RP_ROUTE
    for (uint32_t t = t0; t < t1; ++t) {
        cb_wait_front(cb_k, 1);
        cb_wait_front(cb_s, 1);
        noc_async_write(get_read_ptr(cb_k), s_y.get_noc_addr(t), F32);
        noc_async_write(get_read_ptr(cb_s), s_y.get_noc_addr(n + t), F32);
        noc_async_write_barrier();
        cb_pop_front(cb_k, 1);
        cb_pop_front(cb_s, 1);
    }
#else
    constexpr uint32_t K = 16;              // top-k (ROUTE_WIDE record)
    constexpr uint32_t NC = n;              // cores = key tiles (one each)
    constexpr uint32_t T = RP_ROWS;         // real rows
    constexpr uint32_t G = RP_GROUPS;
    constexpr uint32_t TG = (T + G - 1) / G;    // rows per group (max)
    constexpr uint32_t RS = (TG + NC - 1) / NC;  // rows per owner (max)
    constexpr uint32_t cb_scr = 3;          // [owner row slot][core] lists, then this core's lists, then a record
    constexpr uint32_t LIST_B = 3 * 4 * K;  // keys, ids, s
    constexpr auto tab_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    const uint32_t tab_addr = get_arg_val<uint32_t>(3);
    const uint32_t c = get_arg_val<uint32_t>(4);
    const uint32_t grp = get_arg_val<uint32_t>(5);
    const auto s_tab = TensorAccessor(tab_args, tab_addr, moe_topk::REC_BYTES);
    const uint32_t scr = get_write_ptr(cb_scr);
    const uint32_t own = scr + RS * NC * LIST_B;
    const uint32_t rec = own + TG * LIST_B;
    constexpr uint32_t own_ct = tab_args.next_compile_time_args_offset();
    const uint32_t* gown = &kernel_compile_time_args[own_ct + grp * NC];
    const uint32_t sem = get_semaphore(0);
    volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem);

    cb_wait_front(cb_k, 1);
    cb_wait_front(cb_s, 1);
    const uint32_t kl1 = get_read_ptr(cb_k), sl1 = get_read_ptr(cb_s);
    invalidate_l1_cache();  // (the packer wrote them)
    if (grp == 0) {
        noc_async_write(kl1, s_y.get_noc_addr(t0), F32);
        noc_async_write(sl1, s_y.get_noc_addr(n + t0), F32);
    }
    for (uint32_t r = grp, l = 0; r < T; r += G, ++l) {
        // row r of the FP32 key / s tiles: face 0 (2) columns 0..15 at moe_topk::lg_row_off(r), face 1 (3) + 1024 B
        const uint32_t roff = moe_topk::lg_row_off(r);
        const uint32_t* kf = reinterpret_cast<const uint32_t*>(kl1 + roff);
        const uint32_t* sf = reinterpret_cast<const uint32_t*>(sl1 + roff);
        uint32_t row[32];
        for (uint32_t i = 0; i < 16; ++i) {
            row[i] = kf[i];
            row[16 + i] = kf[256 + i];
        }
        uint32_t keys[K], ids[K];
        moe_topk::select_range_keys<K>(row, 0, 32, keys, ids);  // the gate|up tail's select (ids local 0..31)
        volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(own + l * LIST_B);
        for (uint32_t j = 0; j < K; ++j) {
            const uint32_t i = ids[j];
            o[j] = keys[j];
            o[K + j] = 32 * c + i;
            o[2 * K + j] = sf[i < 16 ? i : 256 + (i - 16)];
        }
        const uint32_t ow = gown[l % NC];
        noc_async_write(own + l * LIST_B, get_noc_addr(ow & 0xFF, (ow >> 8) & 0xFF, scr + ((l / NC) * NC + c) * LIST_B),
                        LIST_B);
    }
    noc_async_write_barrier();  // the lists landed: then the owners' counts
    for (uint32_t r = grp, l = 0; r < T; r += G, ++l) {
        const uint32_t ow = gown[l % NC];
        noc_semaphore_inc(get_noc_addr(ow & 0xFF, (ow >> 8) & 0xFF, sem), 1);
    }
    uint32_t mine = 0;  // the rows this core owns: local rows c, c + NC, ... of its group
    for (uint32_t r = grp + G * c; r < T; r += G * NC) {
        ++mine;
    }
    if (mine) {
        noc_semaphore_wait_min(sem_ptr, NC * mine);
        invalidate_l1_cache();
        for (uint32_t r = grp + G * c, rs = 0; r < T; r += G * NC, ++rs) {
            // NC-way merge of the sorted lists (key descending, ids ascending on ties): the largest head, ties to the
            // lower core (= lower ids) -> the top-k in (key desc, id asc) order, as the tail's select + merge
            const uint32_t base = scr + rs * NC * LIST_B;
            uint32_t hk[NC], hp[NC];
            for (uint32_t q = 0; q < NC; ++q) {
                hp[q] = 0;
                hk[q] = reinterpret_cast<const uint32_t*>(base + q * LIST_B)[0];
            }
            uint32_t ids[K], sv[K];
            for (uint32_t j = 0; j < K; ++j) {
                uint32_t bq = 0, bk = hk[0];
                for (uint32_t q = 1; q < NC; ++q) {
                    if (hk[q] > bk) {
                        bk = hk[q];
                        bq = q;
                    }
                }
                const uint32_t* lq = reinterpret_cast<const uint32_t*>(base + bq * LIST_B);
                const uint32_t h = hp[bq];
                ids[j] = lq[K + h];
                sv[j] = lq[2 * K + h];
                hp[bq] = h + 1;
                hk[bq] = h + 1 < K ? lq[h + 1] : 0;
            }
            // keys_weights (router_topk.hpp) on the winners' s values
            uint64_t s[K];
            uint64_t sum = 0;
            for (uint32_t j = 0; j < K; ++j) {
                const int32_t qv = moe_topk::f32_to_q24(sv[j]);
                s[j] = qv > 0 ? static_cast<uint64_t>(qv) >> 1 : 0;
                sum += s[j];
            }
            if (sum == 0) {
                sum = 1;
            }
            const uint64_t inv = (1ull << 59) / sum;
            volatile tt_l1_ptr uint16_t* rr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rec);
            for (uint32_t j = 0; j < moe_topk::REC_U16; ++j) {
                rr[j] = 0;
            }
            for (uint32_t j = 0; j < K; ++j) {
                uint16_t w = moe_topk::fixed_to_bf16((s[j] * inv) >> 32, 27);
#ifdef RP_WEXP
                if (w) {
                    w = static_cast<uint16_t>(w + (RP_WEXP << 7));
                }
#endif
                rr[j] = static_cast<uint16_t>(ids[j]);
                rr[17 + j] = w;
            }
            rr[16] = 0x3F80;
#ifdef RP_NLIVE
            rr[moe_topk::R_NLIVE] = K;
#endif
            noc_async_write(rec, s_tab.get_noc_addr(r), moe_topk::REC_BYTES);
            noc_async_write_barrier();
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    cb_pop_front(cb_k, 1);
    cb_pop_front(cb_s, 1);
#endif
}
