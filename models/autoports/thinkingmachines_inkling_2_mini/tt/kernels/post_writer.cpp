// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fused decode post-attention section (tt/post.py), writer: the conv's new states, the two cross-core rounds and the
// outputs. A round, one root per row (core t for row t < T): every core writes row t of its P partial tiles (hi / lo
// halves) to row c of root t's gather tiles and increments root t's semaphore; each root's compute reduces them (0 / 1
// matrices) into its totals (row 0), which root t writes into row t of core 0's totals buffer; core 0 multicasts the T
// rows (one multicast per face: concurrent multicasts from several roots serialize) to every core's, then sets
// every core's done semaphore. Start: every core tells core 0 it is running; its go releases the writes.
// POST_MR (T > 1): round 1's two hi / lo pairs of row t reduced by cores 2 t and 2 t + 1 (the same matmuls, half each).
// Signals without round trips: a sender flushes its writes (not a barrier) before the semaphore increments behind
// them (same route, same VC: in order); the totals' multicasts are linked, the done value set by a multicast behind
// them (the matmul in0 sender's scheme); every write is acknowledged before the kernel ends. (An all-to-all of the
// partials, every core multicasting to every core, measured ~150 us per round: concurrent multicasts serialize.)
//
// common args: s0, s1, s2, resid, out, mcast x0, y0, x1, y1, zeros, logits, root t x, y (t < T)

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "red.hpp"
#ifndef RED_ZDEFER
#define RED_ZDEFER 0
#endif
#include "glz.hpp"

// rows 0 .. T-1 of a TH x 32 CB tile (faces min(TH, 16) x 16) into a 32 x 32 tensor tile
template <uint32_t T, uint32_t E, uint32_t TH>
inline void write_rows(uint32_t src, uint64_t dst) {
    constexpr uint32_t FACE = 256 * E;  // (rows 0 .. T-1 of a face are contiguous: one write per face)
    noc_async_write(src, dst, T * 16 * E);
    noc_async_write(src + (TH < 16 ? TH : 16) * 16 * E, dst + FACE, T * 16 * E);
}

void kernel_main() {
    constexpr uint32_t cb_so = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rout = get_compile_time_arg_val(1);
    constexpr uint32_t cb_part = get_compile_time_arg_val(2);
    constexpr uint32_t cb_gath = get_compile_time_arg_val(3);
    constexpr uint32_t cb_part2 = get_compile_time_arg_val(4);
    constexpr uint32_t cb_gath2 = get_compile_time_arg_val(5);
    constexpr uint32_t cb_out = get_compile_time_arg_val(6);
    constexpr uint32_t PER = get_compile_time_arg_val(7);
    constexpr uint32_t C = get_compile_time_arg_val(8);
    constexpr uint32_t NORM = get_compile_time_arg_val(9);
    constexpr uint32_t OUTB = get_compile_time_arg_val(10);
    constexpr uint32_t T = get_compile_time_arg_val(11);
    constexpr uint32_t NQ = get_compile_time_arg_val(12);
    constexpr uint32_t cb_tout = get_compile_time_arg_val(13);  // root: the totals it computed
    constexpr uint32_t cb_tot = get_compile_time_arg_val(14);   // every core: round 1's totals received
    constexpr uint32_t cb_tot2 = get_compile_time_arg_val(15);  // every core: round 2's totals received
    constexpr uint32_t cb_zero = get_compile_time_arg_val(16);
    constexpr uint32_t F32 = 4096;  // a tensor / gather tile
    constexpr uint32_t TH = POST_TH, FT = TH * 128, FO = (TH < 16 ? TH : 16) * 64;  // a CB tile of token rows (fp32)
    constexpr auto s_args = TensorAccessorArgs<17>();
    constexpr auto r_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    constexpr auto z_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();

    const uint32_t c = (uint32_t)get_absolute_logical_y() * POST_CX + (uint32_t)get_absolute_logical_x();
    const uint32_t t0 = c * PER;
    auto ca = [](uint32_t i) { return get_common_arg_val<uint32_t>(i); };
    const auto s0 = TensorAccessor(s_args, ca(0), F32);
    const auto s1 = TensorAccessor(s_args, ca(1), F32);
    const auto s2 = TensorAccessor(s_args, ca(2), F32);
    const auto ro = TensorAccessor(r_args, ca(3), F32);
    const auto oo = TensorAccessor(o_args, ca(4), OUTB);
    const uint32_t mx0 = ca(5), my0 = ca(6), mx1 = ca(7), my1 = ca(8);
    const auto zz = TensorAccessor(z_args, ca(9), 2048);
    auto rxy = [&](uint32_t t, uint32_t a) { return get_noc_addr(ca(11 + 2 * t), ca(12 + 2 * t), a); };
    // root of row c (POST_PACK: core 0 reduces every row, red.hpp; never with the router round)
    const bool root = POST_PACK ? c == 0 : c < T;
    constexpr uint32_t TR = POST_PACK ? T : 1;

    const uint32_t s_ready = get_semaphore(0), s_go = get_semaphore(1), s_g = get_semaphore(2);
    const uint32_t s_done = get_semaphore(3), s_t = get_semaphore(4);
    auto sem = [](uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a); };
    auto mc = [&](uint32_t addr) { return get_noc_multicast_addr(mx0, my0, mx1, my1, addr); };
    auto row_off = [](uint32_t i) { return (i < 16 ? 0u : 2048u) + (i % 16) * 64; };  // (a 32 x 32 tile)

    // receive buffers (the same addresses on every core)
    cb_reserve_back(cb_tot, 2);
    const uint32_t tot = get_write_ptr(cb_tot);
    uint32_t tot2 = 0;
    if constexpr (NORM) {
        cb_reserve_back(cb_tot2, 1);
        tot2 = get_write_ptr(cb_tot2);
    }
    // (every core: the roots' buffer addresses, same CB layout; a round's roots reserve and push their gather)
    const uint32_t g1 = get_write_ptr(cb_gath);
    const uint32_t g2 = NORM ? get_write_ptr(cb_gath2) : 0;
    constexpr uint32_t NP1 = POST_MR && T > 1 ? 2 : 1;  // round 1's roots per row
    {
        DeviceZoneScopedN("P_W_HS");
        noc_semaphore_inc(rxy(0, s_ready), 1);
        if (c == 0) {
            noc_semaphore_wait(sem(s_ready), C);
            *sem(s_go) = 1;
            noc_semaphore_set_multicast(s_go, mc(s_go), C - 1);
        }
        noc_semaphore_wait(sem(s_go), 1);
    GLZ("W_GO");
    }
    // the unused gather rows -> 0 (red.hpp): a zero block (2 KB, read once) copied over them; POST_ZLAZY: round 1's here,
    // the later rounds' after round 1 (off the critical path: a root's zeroing delays its round-1 send)
#if POST_RT
    constexpr uint32_t P3 = 2 * POST_RT;
    const uint32_t g3 = get_write_ptr(POST_CB_G3);
#else
    constexpr uint32_t P3 = 0;
    const uint32_t g3 = 0;
#endif
    constexpr uint32_t PALL = 4 + (NORM ? 2 : 0) + P3;
    uint32_t z = 0, zdone = 0;  // (zdone: the parts zeroed so far)
    auto zero_parts = [&](uint32_t p0, uint32_t p1) {
        for (uint32_t p = p0; p < p1; ++p) {
            if (!(zdone >> p & 1)) {
                red_zero<POST_RK, C, NQ, TR>(z, p < 4 ? g1 + p * NQ * F32 : p < 4 + (NORM ? 2 : 0) ? g2 + (p - 4) * NQ * F32
                                                                    : g3 + (p - 4 - (NORM ? 2 : 0)) * NQ * F32);
                zdone |= 1u << p;
            }
        }
        noc_async_write_barrier();
    };
#if POST_RT && POST_RSPLIT  // router logit tile e of row t reduced by core t ET + e (the root t did all ET tiles)
    constexpr uint32_t PROW = 4 + (NORM ? 2 : 0);  // (the row roots' parts; the router roots zero their own two)
    const bool rr = c < T * POST_RT;
    const uint32_t re = c % POST_RT, rt_ = c / POST_RT;
#else
    constexpr uint32_t PROW = PALL;
    constexpr bool rr = false;
#endif
    if (root || rr || c < T * NP1) {
        cb_reserve_back(cb_zero, 1);
        z = get_write_ptr(cb_zero);
        noc_async_read(zz.get_noc_addr(0), z, 2048);
#if !RED_ZDEFER
        noc_async_read_barrier();
#endif
    }
    // round 1's unused gather rows -> 0 (RED_ZDEFER: after this core's own round-1 send; the zeroed rows are not
    // the ones the partials land in, so only the root's reduce needs them, not the senders)
    auto zero1 = [&]() {
#if RED_ZDEFER
        noc_async_read_barrier();
#endif
        if (NP1 > 1 && c < T * NP1) {  // (its pair's two parts)
            zero_parts(2 * (c % NP1), 2 * (c % NP1) + 2);
        } else if (root) {
            zero_parts(0, POST_ZLAZY ? 4 : PROW);
        }
    };
#if !RED_ZDEFER
    zero1();
#endif

    for (uint32_t i = 0; i < PER; ++i) {
        const uint32_t t = t0 + i;
        cb_wait_front(cb_so, 3);
        const uint32_t so = get_read_ptr(cb_so);
        write_rows<T, 4, TH>(so, s0.get_noc_addr(t));  // (live rows only: the other rows are other slots' / padding)
        write_rows<T, 4, TH>(so + FT, s1.get_noc_addr(t));
        write_rows<T, 4, TH>(so + 2 * FT, s2.get_noc_addr(t));
        cb_wait_front(cb_rout, 1);
        write_rows<T, 4, TH>(get_read_ptr(cb_rout), ro.get_noc_addr(t));
        noc_async_write_barrier();
        cb_pop_front(cb_so, 3);
        cb_pop_front(cb_rout, 1);
    }

    // parts' row t -> row c of root t's gather (rho = c), then root t's semaphore
    // np roots per row: pair p / 2 of row t -> core t np + p / 2
    auto send = [&](uint32_t src, uint32_t parts, uint32_t g, uint32_t np) {
        const uint32_t d = g + red_off<POST_RK, TR>(c);
        for (uint32_t t = 0; t < (POST_PACK ? 1 : T); ++t) {  // (row t < 16 of a CB tile)
            if (POST_PACK) {  // (rows 0 .. T - 1 in one write per face)
                red_send<NQ, FT, FO, T * 64>(src, rxy(0, d), d, 0, parts);
            } else if (np > 1) {
                for (uint32_t j = 0; j < np; ++j) {
                    red_send<NQ, FT, FO>(src + t * 64, rxy(t * np + j, d), d, 2 * j, 2 * j + 2);
                }
            } else {
                red_send<NQ, FT, FO>(src + t * 64, rxy(t, d), d, 0, parts);
            }
        }
        noc_async_writes_flushed();  // (in order with the increments behind them: same route, same VC)
        GLZ("W_S");
        for (uint32_t r = 0; r < (POST_PACK ? 1 : T * np); ++r) {
            noc_semaphore_inc(rxy(r, s_g), 1);
        }
    };
    uint32_t gcnt = 0, tcnt = 0;  // (this core's s_g / s_t targets so far)
    // one round: parts -> the roots, root t's totals (ntot tiles, row 0) -> row t of everyone's
    auto round = [&](uint32_t cb_p, uint32_t parts, uint32_t cb_g, uint32_t g, uint32_t cb_t, uint32_t tb,
                     uint32_t ntot, uint32_t done_val, uint32_t np) {
        cb_wait_front(cb_p, parts);
        GLZ("W_P");
        send(get_read_ptr(cb_p), parts, g, np);
#if RED_ZDEFER
        if (done_val == 1) {
            zero1();
        }
#endif
        cb_pop_front(cb_p, parts);
        if (POST_PACK ? c == 0 : c < T * np) {
            const uint32_t nt = ntot / np, jp = c % np, row = c / np;  // (this root's totals: tiles jp .. jp + nt - 1)
            gcnt += C;
            cb_reserve_back(cb_g, parts * NQ);
            noc_semaphore_wait(sem(s_g), gcnt);
            GLZ("W_G");
            cb_push_back(cb_g, parts * NQ);
            cb_wait_front(cb_tout, nt);
            GLZ("W_T");
            const uint32_t ts = get_read_ptr(cb_tout);
            // the totals' multicasts linked, then the done value set by a multicast behind them (same route / VC:
            // in order, no barrier; the matmul in0 sender's scheme); every core's own copy by a local write
            if constexpr (T == 1 || POST_PACK) {  // (rows 0 .. T - 1)
                for (uint32_t k = 0; k < ntot; ++k) {
                    noc_async_write_multicast(ts + k * FT, mc(tb + k * FT), T * 64, C - 1, true);
                    noc_async_write_multicast(ts + k * FT + FO, mc(tb + k * FT + FO), T * 64, C - 1, true);
                }
                noc_async_writes_flushed();
                *sem(s_done) = done_val;
                noc_semaphore_set_multicast(s_done, mc(s_done), C - 1);
                for (uint32_t k = 0; k < ntot; ++k) {
                    noc_async_write(ts + k * FT, get_noc_addr(tb + k * FT), T * 64);
                    noc_async_write(ts + k * FT + FO, get_noc_addr(tb + k * FT + FO), T * 64);
                }
                noc_async_write_barrier();
            } else {  // row 0 -> row `row` of core 0's totals
                for (uint32_t k = 0; k < nt; ++k) {
                    const uint32_t o = tb + (jp + k) * FT + row * 64;
                    noc_async_write(ts + k * FT, rxy(0, o), 64);
                    noc_async_write(ts + k * FT + FO, rxy(0, o + FO), 64);
                }
                if (c != 0) {
                    noc_async_writes_flushed();
                    noc_semaphore_inc(rxy(0, s_t), 1);
                } else {  // rows 0..T-1 (contiguous per face) -> everyone
                    noc_async_write_barrier();
                    tcnt += T * np - 1;
                    noc_semaphore_wait(sem(s_t), tcnt);
                    for (uint32_t k = 0; k < ntot; ++k) {
                        const uint32_t o = tb + k * FT;
                        noc_async_write_multicast(o, mc(o), T * 64, C - 1, true);
                        noc_async_write_multicast(o + FO, mc(o + FO), T * 64, C - 1, true);
                    }
                    noc_async_writes_flushed();
                    *sem(s_done) = done_val;
                    noc_semaphore_set_multicast(s_done, mc(s_done), C - 1);
                }
            }
            cb_pop_front(cb_tout, nt);
        }
        noc_semaphore_wait(sem(s_done), done_val);
        GLZ("W_D");
        cb_push_back(cb_t, ntot);
    };
    {
        DeviceZoneScopedN("P_W_R1");
        round(cb_part, 4, cb_gath, g1, cb_tot, tot, 2, 1, NP1);
    }
    if ((POST_ZLAZY || NP1 > 1) && root) {
        zero_parts(4, PROW);
    }
#if POST_RT && POST_RSPLIT
    if (rr) {
        zero_parts(PROW + 2 * re, PROW + 2 * re + 2);
    }
#endif
    if constexpr (NORM) {
        round(cb_part2, 2, cb_gath2, g2, cb_tot2, tot2, 1, 2, 1);
    }

    for (uint32_t i = 0; i < PER; ++i) {
        cb_wait_front(cb_out, 1);
        write_rows<T, OUTB / 1024, TH>(get_read_ptr(cb_out), oo.get_noc_addr(t0 + i));
#if POST_DUP
        write_rows<T, OUTB / 1024, TH>(get_read_ptr(cb_out), oo.get_noc_addr(t0 + i) + 16 * (OUTB / 1024));
#endif
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
#if POST_RT
    {  // router round: the partial logits' row t -> root t's gather; root t writes logits row t
        constexpr uint32_t ET = POST_RT, parts = 2 * ET;
        constexpr auto l_args = TensorAccessorArgs<z_args.next_compile_time_args_offset()>();
        cb_wait_front(POST_CB_P3, parts);
#if POST_RSPLIT
        {  // parts 2 e, 2 e + 1 (hi / lo of logit tile e) of row t -> core t ET + e
            const uint32_t src = get_read_ptr(POST_CB_P3), d = get_write_ptr(POST_CB_G3) + red_off<POST_RK>(c);
            for (uint32_t t = 0; t < T; ++t) {
                for (uint32_t p = 0; p < parts; ++p) {
                    const uint32_t s = src + p * FT + t * 64, r = t * ET + p / 2;
                    noc_async_write(s, rxy(r, d + p * NQ * F32), 64);
                    noc_async_write(s + FO, rxy(r, d + p * NQ * F32 + 1024), 64);
                }
            }
            noc_async_writes_flushed();
            for (uint32_t r = 0; r < T * ET; ++r) {
                noc_semaphore_inc(rxy(r, s_g), 1);
            }
        }
        cb_pop_front(POST_CB_P3, parts);
        if (rr) {
            cb_reserve_back(POST_CB_G3, parts * NQ);
            noc_semaphore_wait(sem(s_g), gcnt + C);
            cb_push_back(POST_CB_G3, parts * NQ);
            const auto al = TensorAccessor(l_args, ca(10), F32);
            cb_wait_front(POST_CB_T3, 1);
            const uint64_t dl = al.get_noc_addr(re) + rt_ * 64;
            write_rows<1, 4, TH>(get_read_ptr(POST_CB_T3), dl);
#if POST_DUP
            write_rows<1, 4, TH>(get_read_ptr(POST_CB_T3), dl + 64);
#endif
            noc_async_write_barrier();
            cb_pop_front(POST_CB_T3, 1);
        }
#else
        send(get_read_ptr(POST_CB_P3), parts, get_write_ptr(POST_CB_G3), 1);
        cb_pop_front(POST_CB_P3, parts);
        if (root) {
            cb_reserve_back(POST_CB_G3, parts * NQ);
            noc_semaphore_wait(sem(s_g), gcnt + C);
            cb_push_back(POST_CB_G3, parts * NQ);
            const auto al = TensorAccessor(l_args, ca(10), F32);
            cb_wait_front(POST_CB_T3, ET);
            const uint32_t ts = get_read_ptr(POST_CB_T3);
            for (uint32_t e = 0; e < ET; ++e) {
                const uint64_t dl = al.get_noc_addr(e) + c * 64;
                write_rows<1, 4, TH>(ts + e * FT, dl);
#if POST_DUP
                write_rows<1, 4, TH>(ts + e * FT, dl + 64);
#endif
            }
            noc_async_write_barrier();
            cb_pop_front(POST_CB_T3, ET);
        }
#endif
    }
#endif
    // (the round sends are only flushed: every write and increment acknowledged before the kernel ends)
    GLZ("W_END");
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    *sem(s_ready) = 0;
    *sem(s_go) = 0;
    *sem(s_g) = 0;
    *sem(s_done) = 0;
    *sem(s_t) = 0;
}
