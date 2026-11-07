// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Expert-sorted MoE dispatch bookkeeping on N cores (moe_prefill/dispatch_mc.py). Same outputs, bit for bit, as the
// single-core ttnn.experimental.moe_sorted_dispatch: a stable counting sort of the T*k (token, slot) pairs by expert,
// experts ascending, each expert's rows padded to whole GS groups, then all-padding rows up to RD.
//
// Cores split the tokens (whole 32-token tiles, in order); the experts are split over the first NC cores
// ("collectors", EPC experts each). Every exchange is spread over all cores' NOC endpoints:
//   R1  every core counts its pairs per expert and sends each collector its EPC counts       (sem 0 on collectors)
//   R2  collectors sum their experts' totals to the leader (core 0)                           (sem 1 on the leader)
//   R3  the leader lays out the padded expert offsets + group ids, sends each collector its
//       slice of offsets                                                                       (sem 2 on collectors)
//   R4  collectors turn their [core][expert] counts into each core's first row (cores in token
//       order: the stable order), zero their experts' padding rows, send each core its bases   (sem 3 on every core)
//   R5  every core scatters its pairs into per-expert runs (token order) and writes each run into
//       the owning collector's row buffer, and its undispatch slice to DRAM                    (sem 4 on collectors)
//   R6  every collector writes its experts' contiguous rows to DRAM; the leader writes group ids
//       and the all-padding tail.
// Every core holds the same CB layout, so a remote L1 address equals the local one.
//
// runtime args: [ids, scores, src_tok, weights, group_ids, undispatch addrs, core idx, first tile, tiles,
//                noc x, y of each of the N cores]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t a = 0;
    const uint32_t ids_addr = get_arg_val<uint32_t>(a++);
    const uint32_t sc_addr = get_arg_val<uint32_t>(a++);
    const uint32_t src_addr = get_arg_val<uint32_t>(a++);
    const uint32_t w_addr = get_arg_val<uint32_t>(a++);
    const uint32_t gid_addr = get_arg_val<uint32_t>(a++);
    const uint32_t und_addr = get_arg_val<uint32_t>(a++);
    const uint32_t me = get_arg_val<uint32_t>(a++);
    const uint32_t tile0 = get_arg_val<uint32_t>(a++);
    const uint32_t ntiles = get_arg_val<uint32_t>(a++);
    const uint32_t xy = a;

    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t E = get_compile_time_arg_val(2);
    constexpr uint32_t GS = get_compile_time_arg_val(3);
    constexpr uint32_t RD = get_compile_time_arg_val(4);
    constexpr uint32_t G = get_compile_time_arg_val(5);
    constexpr uint32_t ids_page = get_compile_time_arg_val(6);
    constexpr uint32_t sc_page = get_compile_time_arg_val(7);
    constexpr uint32_t ids_elem_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t pad_group_id = get_compile_time_arg_val(9);
    constexpr uint32_t N = get_compile_time_arg_val(10);
    constexpr uint32_t NC = get_compile_time_arg_val(11);
    constexpr uint32_t EPC = get_compile_time_arg_val(12);  // experts per collector, a multiple of 4 (16-byte slices)
    constexpr uint32_t TPC = get_compile_time_arg_val(13);  // max tiles per core
    constexpr uint32_t cb_in = get_compile_time_arg_val(14);
    constexpr uint32_t cb_und = get_compile_time_arg_val(15);
    constexpr uint32_t cb_vec = get_compile_time_arg_val(16);  // cnt | base | tot | off : 4 x [NC*EPC] uint32
    constexpr uint32_t cb_cntT = get_compile_time_arg_val(17);  // [N][EPC] uint32 (collectors)
    constexpr uint32_t cb_src = get_compile_time_arg_val(18);
    constexpr uint32_t cb_w = get_compile_time_arg_val(19);
    constexpr uint32_t cb_gid = get_compile_time_arg_val(20);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(21);  // eidx | cur_s | cur_w | cur_d : 4 x [E] uint32
    constexpr uint32_t cb_stg = get_compile_time_arg_val(22);
    constexpr uint32_t cb_zero = get_compile_time_arg_val(23);
    constexpr uint32_t STG_SRC_BYTES = get_compile_time_arg_val(24);
    constexpr uint32_t ZERO_BYTES = get_compile_time_arg_val(25);
    constexpr auto ids_args = TensorAccessorArgs<26>();
    constexpr auto sc_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto src_args = TensorAccessorArgs<sc_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto gid_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto und_args = TensorAccessorArgs<gid_args.next_compile_time_args_offset()>();
    constexpr uint32_t EP = NC * EPC;
    constexpr uint32_t SLICE = EPC * 4;

    const auto s_ids = TensorAccessor(ids_args, ids_addr, ids_page);
    const auto s_sc = TensorAccessor(sc_args, sc_addr, sc_page);
    const auto s_src = TensorAccessor(src_args, src_addr, RD * 4);
    const auto s_w = TensorAccessor(w_args, w_addr, RD * 2);
    const auto s_gid = TensorAccessor(gid_args, gid_addr, G * 2);
    const auto s_und = TensorAccessor(und_args, und_addr, T * K * 4);

    const bool leader = me == 0;
    const bool collector = me < NC;
    const uint32_t e0 = me * EPC;
    const uint32_t ids_l1 = get_write_ptr(cb_in);
    const uint32_t sc_l1 = ids_l1 + TPC * ids_page;
    const uint32_t und_l1 = get_write_ptr(cb_und);
    const uint32_t cnt_l1 = get_write_ptr(cb_vec);
    const uint32_t base_l1 = cnt_l1 + EP * 4;
    const uint32_t tot_l1 = base_l1 + EP * 4;
    const uint32_t off_l1 = tot_l1 + EP * 4;
    const uint32_t cntT_l1 = get_write_ptr(cb_cntT);
    const uint32_t src_l1 = get_write_ptr(cb_src);
    const uint32_t w_l1 = get_write_ptr(cb_w);
    const uint32_t gid_l1 = get_write_ptr(cb_gid);
    const uint32_t stg_src = get_write_ptr(cb_stg);
    const uint32_t stg_w = stg_src + STG_SRC_BYTES;
    const uint32_t sem_cnt = get_semaphore(0);
    const uint32_t sem_tot = get_semaphore(1);
    const uint32_t sem_off = get_semaphore(2);
    const uint32_t sem_base = get_semaphore(3);
    const uint32_t sem_runs = get_semaphore(4);
    auto noc_of = [&](uint32_t core, uint32_t addr) -> uint64_t {
        return get_noc_addr(get_arg_val<uint32_t>(xy + 2 * core), get_arg_val<uint32_t>(xy + 2 * core + 1), addr);
    };
    auto wait_reset = [](uint32_t sem, uint32_t n) {
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem);
        noc_semaphore_wait(p, n);
        *p = 0;
        invalidate_l1_cache();
    };

    tt_l1_ptr uint32_t* cnt = reinterpret_cast<tt_l1_ptr uint32_t*>(cnt_l1);
    tt_l1_ptr uint32_t* base = reinterpret_cast<tt_l1_ptr uint32_t*>(base_l1);
    tt_l1_ptr uint32_t* tot = reinterpret_cast<tt_l1_ptr uint32_t*>(tot_l1);
    tt_l1_ptr uint32_t* off = reinterpret_cast<tt_l1_ptr uint32_t*>(off_l1);
    tt_l1_ptr uint32_t* cntT = reinterpret_cast<tt_l1_ptr uint32_t*>(cntT_l1);
    tt_l1_ptr uint32_t* und = reinterpret_cast<tt_l1_ptr uint32_t*>(und_l1);
    tt_l1_ptr uint32_t* src_tok = reinterpret_cast<tt_l1_ptr uint32_t*>(src_l1);
    tt_l1_ptr uint16_t* weights = reinterpret_cast<tt_l1_ptr uint16_t*>(w_l1);
    tt_l1_ptr uint32_t* scr = reinterpret_cast<tt_l1_ptr uint32_t*>(get_write_ptr(cb_scr));
    tt_l1_ptr uint32_t* eidx = scr;
    tt_l1_ptr uint32_t* cur_s = scr + E;
    tt_l1_ptr uint32_t* cur_w = scr + 2 * E;
    tt_l1_ptr uint32_t* cur_d = scr + 3 * E;

    // ---- inputs: this core's top-k tiles
    for (uint32_t i = 0; i < ntiles; ++i) {
        noc_async_read(s_ids.get_noc_addr(tile0 + i), ids_l1 + i * ids_page, ids_page);
        noc_async_read(s_sc.get_noc_addr(tile0 + i), sc_l1 + i * sc_page, sc_page);
    }
    for (uint32_t e = 0; e < EP; ++e) {
        cnt[e] = 0;
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    const tt_l1_ptr uint16_t* ids16 = reinterpret_cast<const tt_l1_ptr uint16_t*>(ids_l1);
    const tt_l1_ptr uint32_t* ids32 = reinterpret_cast<const tt_l1_ptr uint32_t*>(ids_l1);
    const tt_l1_ptr uint16_t* scores = reinterpret_cast<const tt_l1_ptr uint16_t*>(sc_l1);

    // ---- pass 1: decode + count (element (r, c) of a tile: face ((r>>4)<<1 | c>>4), offset (r&15)*16 + (c&15))
    uint32_t npairs = 0;
    for (uint32_t i = 0; i < ntiles; ++i) {
        const uint32_t tile = tile0 + i;
        const uint32_t rows = (tile + 1) * 32 <= T ? 32 : T - tile * 32;
        for (uint32_t rr = 0; rr < rows; ++rr) {
            const uint32_t row_base = i * 1024 + ((rr >> 4) << 9) + ((rr & 15) << 4);
            for (uint32_t j = 0; j < K; ++j, ++npairs) {
                const uint32_t o = row_base + ((j >> 4) << 8) + (j & 15);
                const uint32_t e = ids_elem_bytes == 2 ? static_cast<uint32_t>(ids16[o]) : ids32[o];
                und[npairs] = e;
                if (e < E) {
                    cnt[e] += 1;
                }
            }
        }
    }

    // ---- R1: my counts of each collector's experts -> its [core][expert] table
    asm volatile("fence" ::: "memory");
    for (uint32_t j = 0; j < NC; ++j) {
        noc_async_write(cnt_l1 + j * SLICE, noc_of(j, cntT_l1 + me * SLICE), SLICE);
    }
    noc_async_write_barrier();
    for (uint32_t j = 0; j < NC; ++j) {
        noc_semaphore_inc(noc_of(j, sem_cnt), 1);
    }

    // ---- R2: collectors: totals of my experts -> the leader
    if (collector) {
        wait_reset(sem_cnt, N);
        for (uint32_t i = 0; i < EPC; ++i) {
            uint32_t s = 0;
            for (uint32_t c = 0; c < N; ++c) {
                s += cntT[c * EPC + i];
            }
            tot[e0 + i] = s;
        }
        asm volatile("fence" ::: "memory");
        noc_async_write(tot_l1 + e0 * 4, noc_of(0, tot_l1 + e0 * 4), SLICE);
        noc_async_write_barrier();
        noc_semaphore_inc(noc_of(0, sem_tot), 1);
    }

    // ---- R3: leader: padded offsets, group ids, the all-padding tail
    if (leader) {
        wait_reset(sem_tot, NC);
        tt_l1_ptr uint16_t* group_ids = reinterpret_cast<tt_l1_ptr uint16_t*>(gid_l1);
        uint32_t o = 0;
        for (uint32_t e = 0; e < E; ++e) {
            off[e] = o;
            const uint32_t groups_e = (tot[e] + GS - 1) / GS;
            for (uint32_t g = 0; g < groups_e; ++g) {
                group_ids[o / GS + g] = static_cast<uint16_t>(e);
            }
            o += groups_e * GS;
        }
        for (uint32_t g = o / GS; g < G; ++g) {
            group_ids[g] = static_cast<uint16_t>(pad_group_id);
        }
        asm volatile("fence" ::: "memory");
        for (uint32_t j = 0; j < NC; ++j) {
            noc_async_write(off_l1 + j * SLICE, noc_of(j, off_l1 + j * SLICE), SLICE);
        }
        noc_async_write_barrier();
        for (uint32_t j = 0; j < NC; ++j) {
            noc_semaphore_inc(noc_of(j, sem_off), 1);
        }
        // group ids, and rows [o, RD) are all padding (token 0, weight 0): written from a zeroed block
        noc_async_write(gid_l1, s_gid.get_noc_addr(0), G * 2);
        const uint32_t zl1 = get_write_ptr(cb_zero);
        tt_l1_ptr uint32_t* z = reinterpret_cast<tt_l1_ptr uint32_t*>(zl1);
        for (uint32_t i = 0; i < ZERO_BYTES / 4; ++i) {
            z[i] = 0;
        }
        asm volatile("fence" ::: "memory");
        for (uint32_t b = o * 4; b < RD * 4; b += ZERO_BYTES) {
            noc_async_write(zl1, s_src.get_noc_addr(0, b), RD * 4 - b < ZERO_BYTES ? RD * 4 - b : ZERO_BYTES);
        }
        for (uint32_t b = o * 2; b < RD * 2; b += ZERO_BYTES) {
            noc_async_write(zl1, s_w.get_noc_addr(0, b), RD * 2 - b < ZERO_BYTES ? RD * 2 - b : ZERO_BYTES);
        }
    }

    // ---- R4: collectors: every core's first row per expert, padding rows, bases -> every core
    uint32_t row0 = 0, row1 = 0;  // my collector rows [row0, row1)
    if (collector) {
        wait_reset(sem_off, 1);
        const uint32_t e1 = e0 + EPC < E ? e0 + EPC : E;
        for (uint32_t e = e0; e < e1; ++e) {
            const uint32_t i = e - e0;
            uint32_t run = off[e];
            for (uint32_t c = 0; c < N; ++c) {
                const uint32_t b = run;
                run += cntT[c * EPC + i];
                cntT[c * EPC + i] = b;
            }
            const uint32_t end = off[e] + (tot[e] + GS - 1) / GS * GS;
            for (uint32_t d = run; d < end; ++d) {
                src_tok[d] = 0;
                weights[d] = 0;
            }
            if (e == e0) {
                row0 = off[e];
            }
            row1 = end;
        }
        asm volatile("fence" ::: "memory");
        for (uint32_t c = 0; c < N; ++c) {
            noc_async_write(cntT_l1 + c * SLICE, noc_of(c, base_l1 + e0 * 4), SLICE);
        }
        noc_async_write_barrier();
        for (uint32_t c = 0; c < N; ++c) {
            noc_semaphore_inc(noc_of(c, sem_base), 1);
        }
    }

    // ---- R5: my pairs -> per-expert runs (token order) -> the owning collector's row buffers
    wait_reset(sem_base, NC);
    {
        // staging: a run starts at the same address mod 16 as its destination (src_l1 / w_l1 + row offset)
        uint32_t ps = stg_src, pw = stg_w;
        for (uint32_t e = 0; e < E; ++e) {
            if (cnt[e]) {
                const uint32_t b = base[e];
                ps = ((ps + 15) & ~15u) + ((src_l1 + 4 * b) & 15);
                pw = ((pw + 15) & ~15u) + ((w_l1 + 2 * b) & 15);
                cur_s[e] = ps;
                cur_w[e] = pw;
                cur_d[e] = b;
                ps += 4 * cnt[e];
                pw += 2 * cnt[e];
            }
        }
    }
    {
        uint32_t i = 0;
        for (uint32_t ti = 0; ti < ntiles; ++ti) {
            const uint32_t tile = tile0 + ti;
            const uint32_t rows = (tile + 1) * 32 <= T ? 32 : T - tile * 32;
            for (uint32_t rr = 0; rr < rows; ++rr) {
                const uint32_t t = tile * 32 + rr;
                const uint32_t row_base = ti * 1024 + ((rr >> 4) << 9) + ((rr & 15) << 4);
                for (uint32_t j = 0; j < K; ++j, ++i) {
                    const uint32_t e = und[i];
                    if (e < E) {
                        *reinterpret_cast<tt_l1_ptr uint32_t*>(cur_s[e]) = t;
                        cur_s[e] += 4;
                        *reinterpret_cast<tt_l1_ptr uint16_t*>(cur_w[e]) = scores[row_base + ((j >> 4) << 8) + (j & 15)];
                        cur_w[e] += 2;
                        und[i] = cur_d[e]++;
                    } else {
                        und[i] = RD - GS;  // sentinel pairs land in the trailing padding group (never read back)
                    }
                }
            }
        }
    }
    asm volatile("fence" ::: "memory");
    for (uint32_t e = 0; e < E; ++e) {
        const uint32_t n = cnt[e];
        if (n) {
            const uint32_t b = base[e];
            const uint32_t owner = e / EPC;
            noc_async_write(cur_s[e] - 4 * n, noc_of(owner, src_l1 + 4 * b), 4 * n);
            noc_async_write(cur_w[e] - 2 * n, noc_of(owner, w_l1 + 2 * b), 2 * n);
        }
    }
    noc_async_write(und_l1, s_und.get_noc_addr(0, tile0 * 32 * K * 4), npairs * 4);
    noc_async_write_barrier();
    for (uint32_t j = 0; j < NC; ++j) {
        noc_semaphore_inc(noc_of(j, sem_runs), 1);
    }

    // ---- R6: collectors: my experts' rows -> DRAM
    if (collector) {
        wait_reset(sem_runs, N);
        if (row1 > row0) {
            noc_async_write(src_l1 + 4 * row0, s_src.get_noc_addr(0, 4 * row0), 4 * (row1 - row0));
            noc_async_write(w_l1 + 2 * row0, s_w.get_noc_addr(0, 2 * row0), 2 * (row1 - row0));
        }
    }
    noc_async_write_barrier();
}
