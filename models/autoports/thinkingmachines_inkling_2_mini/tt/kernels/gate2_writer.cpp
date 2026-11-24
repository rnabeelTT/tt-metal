// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Message gate, round-reduced (tt/gate.py GateKernel2), writer: the cross-core rounds (tt/post.py's scheme: a root per
// row, core t: row t of the partials -> row c of root t's gather tiles; root t's totals (row 0) -> row t of core 0's,
// multicast by core 0) and the output's live rows. Every round uses the same gather buffer (GP parts of NSEL tiles): a
// core sends round k + 1 only after round k's totals, which each root computes after consuming round k's gather.
//   rounds: 1 sum h^2 / m^2 (4 parts), 2 the partial down projections (4); NORM: 3 sum out^2 (2), the output is the
//   normed bf16 out; ROUTER: 4 the partial router logits (2 ET parts), root t writes logits row t (no broadcast).
//   DUP (T = 1): row 0 of the output / logits also into row 1 (the MoE's two-row path).
//   G2_MR (T > 1): rounds 1 / 2's hi / lo pair j of row t reduced by core 2 t + j (half the root matmuls each).
//
// common args: out, logits, mcast x0, y0, x1, y1, zeros, root t x, y (t < T)

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "red.hpp"

// rows 0 .. T-1 of a TH x 32 CB tile (faces min(TH, 16) x 16) into a 32 x 32 tensor tile
template <uint32_t T, uint32_t E, uint32_t TH>
inline void write_rows(uint32_t src, uint64_t dst) {
    constexpr uint32_t FACE = 256 * E;  // (rows 0 .. T-1 of a face are contiguous: one write per face)
    noc_async_write(src, dst, T * 16 * E);
    noc_async_write(src + (TH < 16 ? TH : 16) * 16 * E, dst + FACE, T * 16 * E);
}

void kernel_main() {
    constexpr uint32_t cb_part = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gath = get_compile_time_arg_val(1);
    constexpr uint32_t cb_tout = get_compile_time_arg_val(2);
    constexpr uint32_t cb_tot = get_compile_time_arg_val(3);
    constexpr uint32_t cb_tot2 = get_compile_time_arg_val(4);
    constexpr uint32_t cb_tot3 = get_compile_time_arg_val(5);
    constexpr uint32_t cb_out = get_compile_time_arg_val(6);
    constexpr uint32_t cb_zero = get_compile_time_arg_val(7);
    constexpr uint32_t PER = get_compile_time_arg_val(8);
    constexpr uint32_t C = get_compile_time_arg_val(9);
    constexpr uint32_t NSEL = get_compile_time_arg_val(10);
    constexpr uint32_t GP = get_compile_time_arg_val(11);  // gather parts (the widest round)
    constexpr uint32_t NORM = get_compile_time_arg_val(12);
    constexpr uint32_t ROUTER = get_compile_time_arg_val(13);
    constexpr uint32_t ET = get_compile_time_arg_val(14);
    constexpr uint32_t DUP = get_compile_time_arg_val(15);
    constexpr uint32_t T = G2_ROWS;
    constexpr uint32_t F32 = 4096;  // a tensor / gather tile
    constexpr uint32_t TH = G2_TH, FT = TH * 128, FO = (TH < 16 ? TH : 16) * 64;  // a CB tile of token rows (fp32)
    constexpr uint32_t OE = NORM ? 2 : 4;  // output element bytes
    constexpr auto o_args = TensorAccessorArgs<16>();
    constexpr auto l_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto z_args = TensorAccessorArgs<l_args.next_compile_time_args_offset()>();

#ifndef G2_X0
#define G2_X0 0
#define G2_Y0 0
#endif
    // this core's slot in its rectangle (origin G2_X0, G2_Y0; GateKernel2.multi runs several rectangles)
    const uint32_t c = ((uint32_t)get_absolute_logical_y() - G2_Y0) * G2_CX + (uint32_t)get_absolute_logical_x() - G2_X0;
    const uint32_t t0 = c * PER;
    // root of row c (G2_PACK: core 0 reduces every row, red.hpp; never with the router round)
    const bool root = G2_PACK ? c == 0 : c < T;
    constexpr uint32_t TR = G2_PACK ? T : 1;
    auto ca = [](uint32_t i) { return get_common_arg_val<uint32_t>(i); };
    const auto ao = TensorAccessor(o_args, ca(0), 1024 * OE);
    const uint32_t mx0 = ca(2), my0 = ca(3), mx1 = ca(4), my1 = ca(5);
    const auto zz = TensorAccessor(z_args, ca(6), 2048);
    auto rxy = [&](uint32_t t, uint32_t a) { return get_noc_addr(ca(7 + 2 * t), ca(8 + 2 * t), a); };

    const uint32_t s_ready = get_semaphore(0), s_go = get_semaphore(1), s_g = get_semaphore(2);
    const uint32_t s_done = get_semaphore(3), s_t = get_semaphore(4);
    auto sem = [](uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a); };
    auto mc = [&](uint32_t addr) { return get_noc_multicast_addr(mx0, my0, mx1, my1, addr); };
    auto row_off = [](uint32_t i) { return (i < 16 ? 0u : 2048u) + (i % 16) * 64; };
    constexpr uint32_t GT = GP * NSEL;

    cb_reserve_back(cb_tot, 2);
    const uint32_t tot = get_write_ptr(cb_tot);
    cb_reserve_back(cb_tot2, 2);
    const uint32_t tot2 = get_write_ptr(cb_tot2);
    uint32_t tot3 = 0;
    if constexpr (NORM) {
        cb_reserve_back(cb_tot3, 1);
        tot3 = get_write_ptr(cb_tot3);
    }
    const uint32_t g = get_write_ptr(cb_gath);  // (the base: every round wraps back to it)
    {
    DeviceZoneScopedN("G2_W_HS");
    noc_semaphore_inc(rxy(0, s_ready), 1);
    if (c == 0) {
        noc_semaphore_wait(sem(s_ready), C);
        *sem(s_go) = 1;
        noc_semaphore_set_multicast(s_go, mc(s_go), C - 1);
    }
    noc_semaphore_wait(sem(s_go), 1);
    }
    // the unused gather rows -> 0 (red.hpp); G2_ZLAZY: round 1's parts here, the later rounds' after round 1
    constexpr uint32_t NP = G2_MR && T > 1 ? 2 : 1;  // rounds 1 / 2's roots per row
    uint32_t z = 0, zdone = 0;  // (zdone: the parts zeroed so far)
    auto zero_parts = [&](uint32_t p0, uint32_t p1) {
        for (uint32_t p = p0; p < p1; ++p) {
            if (!(zdone >> p & 1)) {
                red_zero<G2_RK, C, NSEL, TR>(z, g + p * NSEL * F32);
                zdone |= 1u << p;
            }
        }
        noc_async_write_barrier();
    };
#if G2_RSPLIT  // router logit tile e of row t reduced by core t ET + e (the root t did all ET tiles)
    const bool rr = ROUTER && c < T * ET;
    const uint32_t re = c % ET, rt_ = c / ET;
#else
    constexpr bool rr = false;
    constexpr uint32_t re = 0, rt_ = 0;
#endif
    if (root || rr || c < T * NP) {
        cb_reserve_back(cb_zero, 1);
        z = get_write_ptr(cb_zero);
        noc_async_read(zz.get_noc_addr(0), z, 2048);
        noc_async_read_barrier();
    }
    if (NP > 1 && c < T * NP) {  // (its pair's two parts)
        zero_parts(2 * (c % NP), 2 * (c % NP) + 2);
    } else if (root) {
        zero_parts(0, G2_ZLAZY ? 4 : GP);
    }
    uint32_t gcnt = 0, tcnt = 0;  // (this core's s_g / s_t targets so far)

    // one round: parts -> the roots; with a broadcast, the roots' ntot totals -> everyone's tb
    // np roots per row: pair p / 2 of row t -> core t np + p / 2
    auto round = [&](uint32_t parts, uint32_t cb_t, uint32_t tb, uint32_t k, bool bcast, uint32_t np) {
        cb_wait_front(cb_part, parts);
        const uint32_t src = get_read_ptr(cb_part);
        const uint32_t d = g + red_off<G2_RK, TR>(c);
        for (uint32_t t = 0; t < (G2_PACK ? 1 : T); ++t) {  // (row t < 16 of a CB tile)
            if (G2_PACK) {  // (rows 0 .. T - 1 in one write per face)
                red_send<NSEL, FT, FO, T * 64>(src, rxy(0, d), d, 0, parts);
            } else if (np > 1) {
                for (uint32_t j = 0; j < np; ++j) {
                    red_send<NSEL, FT, FO>(src + t * 64, rxy(t * np + j, d), d, 2 * j, 2 * j + 2);
                }
            } else {
                red_send<NSEL, FT, FO>(src + t * 64, rxy(t, d), d, 0, parts);
            }
        }
        noc_async_writes_flushed();  // (in order with the increments behind them: same route, same VC)
        for (uint32_t r = 0; r < (G2_PACK ? 1 : T * np); ++r) {
            noc_semaphore_inc(rxy(r, s_g), 1);
        }
        cb_pop_front(cb_part, parts);
        const bool isr = G2_PACK ? c == 0 : c < T * np;
        if (isr) {
            gcnt += C;
            cb_reserve_back(cb_gath, GT);
            noc_semaphore_wait(sem(s_g), gcnt);
            cb_push_back(cb_gath, GT);
        }
        if (!bcast) {
            return;
        }
        const uint32_t ntot = parts / 2, nt = ntot / np, jp = c % np, row = c / np;  // (this root's: tiles jp .. jp + nt - 1)
        if (isr) {
            cb_wait_front(cb_tout, nt);
            const uint32_t ts = get_read_ptr(cb_tout);
            // the totals' multicasts linked, then the done value set by a multicast behind them (same route / VC:
            // in order, no barrier; the matmul in0 sender's scheme); every core's own copy by a local write
            if constexpr (T == 1 || G2_PACK) {  // (rows 0 .. T - 1)
                for (uint32_t j = 0; j < ntot; ++j) {
                    noc_async_write_multicast(ts + j * FT, mc(tb + j * FT), T * 64, C - 1, true);
                    noc_async_write_multicast(ts + j * FT + FO, mc(tb + j * FT + FO), T * 64, C - 1, true);
                }
                noc_async_writes_flushed();
                *sem(s_done) = k;
                noc_semaphore_set_multicast(s_done, mc(s_done), C - 1);
                for (uint32_t j = 0; j < ntot; ++j) {
                    noc_async_write(ts + j * FT, get_noc_addr(tb + j * FT), T * 64);
                    noc_async_write(ts + j * FT + FO, get_noc_addr(tb + j * FT + FO), T * 64);
                }
                noc_async_write_barrier();
            } else {  // row 0 -> row `row` of core 0's totals
                for (uint32_t i = 0; i < nt; ++i) {
                    const uint32_t o = tb + (jp + i) * FT + row * 64;
                    noc_async_write(ts + i * FT, rxy(0, o), 64);
                    noc_async_write(ts + i * FT + FO, rxy(0, o + FO), 64);
                }
                if (c != 0) {
                    noc_async_writes_flushed();
                    noc_semaphore_inc(rxy(0, s_t), 1);
                } else {  // rows 0..T-1 (contiguous per face) -> everyone (one multicaster: concurrent ones serialize)
                    noc_async_write_barrier();
                    tcnt += T * np - 1;
                    noc_semaphore_wait(sem(s_t), tcnt);
                    for (uint32_t j = 0; j < ntot; ++j) {
                        const uint32_t o = tb + j * FT;
                        noc_async_write_multicast(o, mc(o), T * 64, C - 1, true);
                        noc_async_write_multicast(o + FO, mc(o + FO), T * 64, C - 1, true);
                    }
                    noc_async_writes_flushed();
                    *sem(s_done) = k;
                    noc_semaphore_set_multicast(s_done, mc(s_done), C - 1);
                }
            }
            cb_pop_front(cb_tout, nt);
        }
        noc_semaphore_wait(sem(s_done), k);
        cb_push_back(cb_t, ntot);
    };
    {
    DeviceZoneScopedN("G2_W_R1");
    round(4, cb_tot, tot, 1, true, NP);
    }
    if ((G2_ZLAZY || NP > 1) && root) {  // (mr: round 3's parts 0, 1; the router round's)
        zero_parts(0, NP > 1 && G2_RSPLIT ? 2 : GP);
    }
    if (rr) {  // (its two router parts)
        zero_parts(2 * re, 2 * re + 2);
    }
    {
    DeviceZoneScopedN("G2_W_R2");
    round(4, cb_tot2, tot2, 2, true, NP);
    }
    if constexpr (NORM) {
        DeviceZoneScopedN("G2_W_R3");
        round(2, cb_tot3, tot3, 3, true, 1);
    }
    for (uint32_t i = 0; i < PER; ++i) {
        cb_wait_front(cb_out, 1);
        const uint32_t s = get_read_ptr(cb_out);
        const uint64_t d = ao.get_noc_addr(t0 + i);
        write_rows<T, OE, TH>(s, d);
        if constexpr (DUP) {
            write_rows<T, OE, TH>(s, d + 16 * OE);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
    if constexpr (ROUTER) {
        DeviceZoneScopedN("G2_W_R4");
#if G2_RSPLIT
        cb_wait_front(cb_part, 2 * ET);
        {  // parts 2 e, 2 e + 1 (hi / lo of logit tile e) of row t -> core t ET + e
            const uint32_t src = get_read_ptr(cb_part), d = g + red_off<G2_RK>(c);
            for (uint32_t t = 0; t < T; ++t) {
                for (uint32_t p = 0; p < 2 * ET; ++p) {
                    const uint32_t s = src + p * FT + t * 64, r = t * ET + p / 2;
                    noc_async_write(s, rxy(r, d + p * NSEL * F32), 64);
                    noc_async_write(s + FO, rxy(r, d + p * NSEL * F32 + 1024), 64);
                }
            }
            noc_async_writes_flushed();
            for (uint32_t r = 0; r < T * ET; ++r) {
                noc_semaphore_inc(rxy(r, s_g), 1);
            }
        }
        cb_pop_front(cb_part, 2 * ET);
        if (rr) {
            cb_reserve_back(cb_gath, GT);
            noc_semaphore_wait(sem(s_g), gcnt + C);
            cb_push_back(cb_gath, GT);
            const auto al = TensorAccessor(l_args, ca(1), F32);
            cb_wait_front(cb_tout, 1);
            const uint64_t d = al.get_noc_addr(re) + rt_ * 64;
            write_rows<1, 4, TH>(get_read_ptr(cb_tout), d);
            if constexpr (DUP) {
                write_rows<1, 4, TH>(get_read_ptr(cb_tout), d + 64);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_tout, 1);
        }
        if (false) {
#else
        round(2 * ET, 0, 0, 4, false, 1);
        if (root) {  // logits row c ([Tp, E] fp32)
#endif
            const auto al = TensorAccessor(l_args, ca(1), F32);
            cb_wait_front(cb_tout, ET);
            const uint32_t ts = get_read_ptr(cb_tout);
            for (uint32_t e = 0; e < ET; ++e) {
                const uint64_t d = al.get_noc_addr(e) + c * 64;
                write_rows<1, 4, TH>(ts + e * FT, d);
                if constexpr (DUP) {
                    write_rows<1, 4, TH>(ts + e * FT, d + 64);
                }
            }
            noc_async_write_barrier();
            cb_pop_front(cb_tout, ET);
        }
    }
    // (the round sends are only flushed: every write and increment acknowledged before the kernel ends)
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    *sem(s_ready) = 0;
    *sem(s_go) = 0;
    *sem(s_g) = 0;
    *sem(s_done) = 0;
    *sem(s_t) = 0;
}
