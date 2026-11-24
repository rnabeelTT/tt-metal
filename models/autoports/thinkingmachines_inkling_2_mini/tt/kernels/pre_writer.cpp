// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fused decode pre-attention section (tt/pre.py), writer: the conv's states and x (TAIL), the two cross-core rounds
// (the post-attention writer's scheme: a root per row, core t: row t of the partials -> row c of root t's gather
// tiles, root t's totals (row 0) -> row t of core 0's, multicast by core 0), xn (BOUNDARY) and the NQ outputs. Live
// rows only. PRE_NP1 / PRE_NP2 > 1 (T > 1): a round's hi / lo pair j of row t reduced by core t NP + j.
//
// common args: s0, s1, s2, x_out, xn, out[NQ], mcast x0, y0, x1, y1, zeros, root t x, y (t < T)

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "red.hpp"

// rows 0 .. T-1 of a TH x 32 CB tile (faces min(TH, 16) x 16) into a 32 x 32 tensor tile
template <uint32_t T, uint32_t E, uint32_t TH>
inline void write_rows(uint32_t src, uint64_t dst) {
    constexpr uint32_t FACE = 256 * E;  // (rows 0 .. T-1 of a face are contiguous: one write per face)
    noc_async_write(src, dst, T * 16 * E);
    noc_async_write(src + (TH < 16 ? TH : 16) * 16 * E, dst + FACE, T * 16 * E);
}

void kernel_main() {
    constexpr uint32_t cb_so = get_compile_time_arg_val(0);
    constexpr uint32_t cb_xo = get_compile_time_arg_val(1);
    constexpr uint32_t cb_part = get_compile_time_arg_val(2);
    constexpr uint32_t cb_gath = get_compile_time_arg_val(3);
    constexpr uint32_t cb_tout = get_compile_time_arg_val(4);
    constexpr uint32_t cb_tot = get_compile_time_arg_val(5);
    constexpr uint32_t cb_xn = get_compile_time_arg_val(6);
    constexpr uint32_t cb_part2 = get_compile_time_arg_val(7);
    constexpr uint32_t cb_gath2 = get_compile_time_arg_val(8);
    constexpr uint32_t cb_tot2 = get_compile_time_arg_val(9);
    constexpr uint32_t cb_out = get_compile_time_arg_val(10);
    constexpr uint32_t cb_zero = get_compile_time_arg_val(11);
    constexpr uint32_t PER = get_compile_time_arg_val(12);
    constexpr uint32_t C = get_compile_time_arg_val(13);
    constexpr uint32_t NQ = get_compile_time_arg_val(14);
    constexpr uint32_t TAIL = get_compile_time_arg_val(15);
    constexpr uint32_t NORM = get_compile_time_arg_val(16);
    constexpr uint32_t BOUND = get_compile_time_arg_val(17);
    constexpr uint32_t NSEL = get_compile_time_arg_val(18);
    constexpr uint32_t T = PRE_ROWS;
    constexpr uint32_t F32 = 4096;  // a tensor / gather tile
    constexpr uint32_t TH = PRE_TH, FT = TH * 128, FO = (TH < 16 ? TH : 16) * 64;  // a CB tile of token rows (fp32)
    constexpr uint32_t OE = NORM ? 2 : 4;  // output element bytes
    constexpr uint32_t K1 = 1 + NQ;
    constexpr auto s_args = TensorAccessorArgs<19>();                                     // states
    constexpr auto x_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();  // x out, xn, outs
    constexpr auto z_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();  // zeros

    const uint32_t c = (uint32_t)get_absolute_logical_y() * PRE_CX + (uint32_t)get_absolute_logical_x();
    const uint32_t t0 = c * PER;
    const bool root = c < T;  // root of row c
    auto ca = [](uint32_t i) { return get_common_arg_val<uint32_t>(i); };
    constexpr uint32_t am = 5 + NQ;
    const uint32_t mx0 = ca(am), my0 = ca(am + 1), mx1 = ca(am + 2), my1 = ca(am + 3);
    const auto zz = TensorAccessor(z_args, ca(am + 4), 2048);
    auto rxy = [&](uint32_t t, uint32_t a) { return get_noc_addr(ca(am + 5 + 2 * t), ca(am + 6 + 2 * t), a); };

    const uint32_t s_ready = get_semaphore(0), s_go = get_semaphore(1), s_g = get_semaphore(2);
    const uint32_t s_done = get_semaphore(3), s_t = get_semaphore(4);
    auto sem = [](uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a); };
    auto mc = [&](uint32_t addr) { return get_noc_multicast_addr(mx0, my0, mx1, my1, addr); };
    auto row_off = [](uint32_t i) { return (i < 16 ? 0u : 2048u) + (i % 16) * 64; };

    cb_reserve_back(cb_tot, K1);
    const uint32_t tot = get_write_ptr(cb_tot);
    uint32_t tot2 = 0;
    if constexpr (NORM) {
        cb_reserve_back(cb_tot2, NQ);
        tot2 = get_write_ptr(cb_tot2);
    }
    // one gather buffer for both rounds (2 K1 parts of NSEL tiles; round 2's 2 NQ parts are its first ones): each
    // round's root reserves / pushes it whole, its compute pops it whole, so every round starts at the base
    constexpr uint32_t GT = 2 * K1 * NSEL;
    const uint32_t g1 = get_write_ptr(cb_gath);
    noc_semaphore_inc(rxy(0, s_ready), 1);
    if (c == 0) {
        noc_semaphore_wait(sem(s_ready), C);
        *sem(s_go) = 1;
        noc_semaphore_set_multicast(s_go, mc(s_go), C - 1);
    }
    noc_semaphore_wait(sem(s_go), 1);
    // the unused gather rows -> 0 (red.hpp): a round's root zeroes the parts it reduces (zdone: those zeroed so far)
    constexpr uint32_t NP1 = PRE_NP1, NP2 = PRE_NP2;
    constexpr uint32_t TR = PRE_PACK ? T : 1;  // (PRE_PACK: core 0 reduces every row, red.hpp)
    auto isroot = [&](uint32_t np) { return PRE_PACK ? c == 0 : c < T * np; };
    uint32_t z = 0, zdone = 0;
    auto zero_parts = [&](uint32_t p0, uint32_t p1) {
        for (uint32_t p = p0; p < p1; ++p) {
            if (!(zdone >> p & 1)) {
                red_zero<PRE_RK, C, NSEL, TR>(z, g1 + p * NSEL * F32);
                zdone |= 1u << p;
            }
        }
        noc_async_write_barrier();
    };
    if (isroot(NP1) || (NORM && isroot(NP2))) {
        cb_reserve_back(cb_zero, 1);
        z = get_write_ptr(cb_zero);
        noc_async_read(zz.get_noc_addr(0), z, 2048);
        noc_async_read_barrier();
    }
    if (NP1 > 1 && c < T * NP1) {
        zero_parts(2 * (c % NP1), 2 * (c % NP1) + 2);
    } else if (isroot(1)) {
        zero_parts(0, 2 * K1);
    }

    if constexpr (TAIL) {
        const auto x_o = TensorAccessor(x_args, ca(3), 2048);
        for (uint32_t i = 0; i < PER; ++i) {
            const uint32_t t = t0 + i;
            cb_wait_front(cb_so, 3);
            const uint32_t so = get_read_ptr(cb_so);
            for (uint32_t j = 0; j < 3; ++j) {
                const auto as = TensorAccessor(s_args, ca(j), F32);
                write_rows<T, 4, TH>(so + j * FT, as.get_noc_addr(t));
            }
            cb_wait_front(cb_xo, 1);
            write_rows<T, 2, TH>(get_read_ptr(cb_xo), x_o.get_noc_addr(t));
            noc_async_write_barrier();
            cb_pop_front(cb_so, 3);
            cb_pop_front(cb_xo, 1);
        }
    }

    uint32_t gcnt = 0, tcnt = 0;  // (this core's s_g / s_t targets so far)
    // np roots per row (1 or every pair): pair p / 2 of row t -> core t np + p / 2
    auto round = [&](uint32_t cb_p, uint32_t parts, uint32_t cb_g, uint32_t g, uint32_t cb_t, uint32_t tb,
                     uint32_t done_val, uint32_t np) {
        cb_wait_front(cb_p, parts);
        const uint32_t src = get_read_ptr(cb_p);
        const uint32_t d = g + red_off<PRE_RK, TR>(c);
        for (uint32_t t = 0; t < (PRE_PACK ? 1 : T); ++t) {  // (row t < 16 of a CB tile)
            if (PRE_PACK) {  // (rows 0 .. T - 1 in one write per face)
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
        for (uint32_t r = 0; r < (PRE_PACK ? 1 : T * np); ++r) {
            noc_semaphore_inc(rxy(r, s_g), 1);
        }
        cb_pop_front(cb_p, parts);
        const uint32_t ntot = parts / 2, nt = ntot / np, jp = c % np, row = c / np;  // (this root's: tiles jp ..)
        if (isroot(np)) {
            gcnt += C;
            cb_reserve_back(cb_g, GT);
            noc_semaphore_wait(sem(s_g), gcnt);
            cb_push_back(cb_g, GT);
            cb_wait_front(cb_tout, nt);
            const uint32_t ts = get_read_ptr(cb_tout);
            // the totals' multicasts linked, then the done value set by a multicast behind them (same route / VC:
            // in order, no barrier; the matmul in0 sender's scheme); every core's own copy by a local write
            if constexpr (T == 1 || PRE_PACK) {  // (rows 0 .. T - 1)
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
                } else {  // rows 0..T-1 (contiguous per face) -> everyone (one multicaster: concurrent ones serialize)
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
        cb_push_back(cb_t, ntot);
    };
    round(cb_part, 2 * K1, cb_gath, g1, cb_tot, tot, 1, NP1);

    if constexpr (NORM) {
        if (isroot(NP2)) {  // (round 2's parts)
            zero_parts(NP2 > 1 ? 2 * (c % NP2) : 0, NP2 > 1 ? 2 * (c % NP2) + 2 : 2 * NQ);
        }
        round(cb_part2, 2 * NQ, cb_gath, g1, cb_tot2, tot2, 2, NP2);
    } else {
        for (uint32_t q = 0; q < NQ; ++q) {
            const auto ao = TensorAccessor(x_args, ca(5 + q), 1024 * OE);
            for (uint32_t i = 0; i < PER; ++i) {
                cb_wait_front(cb_out, 1);
                write_rows<T, OE, TH>(get_read_ptr(cb_out), ao.get_noc_addr(t0 + i));
                noc_async_write_barrier();
                cb_pop_front(cb_out, 1);
            }
        }
    }
    if constexpr (BOUND) {
        const auto xn = TensorAccessor(x_args, ca(4), F32);
        for (uint32_t i = 0; i < PER; ++i) {
            cb_wait_front(cb_xn, 1);
            write_rows<T, 4, TH>(get_read_ptr(cb_xn), xn.get_noc_addr(t0 + i));
            noc_async_write_barrier();
            cb_pop_front(cb_xn, 1);
        }
    }
    if constexpr (NORM) {
        for (uint32_t q = 0; q < NQ; ++q) {
            const auto ao = TensorAccessor(x_args, ca(5 + q), 1024 * OE);
            for (uint32_t i = 0; i < PER; ++i) {
                cb_wait_front(cb_out, 1);
                write_rows<T, OE, TH>(get_read_ptr(cb_out), ao.get_noc_addr(t0 + i));
                noc_async_write_barrier();
                cb_pop_front(cb_out, 1);
            }
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
