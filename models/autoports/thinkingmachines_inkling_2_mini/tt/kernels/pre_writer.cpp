// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fused decode pre-attention section (tt/pre.py), writer: the conv's states and x (TAIL), the two cross-core rounds
// (the post-attention writer's scheme: live partial rows -> the root's gather tiles, the root's totals multicast
// back), xn (BOUNDARY) and the NQ outputs. Live rows only.
//
// common args: s0, s1, s2, x_out, xn, out[NQ], mcast x0, y0, x1, y1, zeros

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

template <uint32_t T, uint32_t E>
inline void write_rows(uint32_t src, uint64_t dst) {
    constexpr uint32_t HALF = 16 * E, FACE = 256 * E;
    for (uint32_t r = 0; r < T; ++r) {
        noc_async_write(src + r * HALF, dst + r * HALF, HALF);
        noc_async_write(src + FACE + r * HALF, dst + FACE + r * HALF, HALF);
    }
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
    constexpr uint32_t F32 = 4096;
    constexpr uint32_t OE = NORM ? 2 : 4;  // output element bytes
    constexpr uint32_t K1 = 1 + NQ;
    constexpr auto s_args = TensorAccessorArgs<19>();                                     // states
    constexpr auto x_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();  // x out, xn, outs
    constexpr auto z_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();  // zeros

    const uint32_t c = (uint32_t)get_absolute_logical_y() * PRE_CX + (uint32_t)get_absolute_logical_x();
    const uint32_t t0 = c * PER;
    const bool root = c == 0;
    auto ca = [](uint32_t i) { return get_common_arg_val<uint32_t>(i); };
    constexpr uint32_t am = 5 + NQ;
    const uint32_t mx0 = ca(am), my0 = ca(am + 1), mx1 = ca(am + 2), my1 = ca(am + 3);
    const auto zz = TensorAccessor(z_args, ca(am + 4), 2048);
    const uint32_t rx = mx0, ry = my0;  // the root: the rectangle's first core

    const uint32_t s_ready = get_semaphore(0), s_go = get_semaphore(1), s_g = get_semaphore(2);
    const uint32_t s_done = get_semaphore(3);
    auto sem = [](uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a); };
    auto mc = [&](uint32_t addr) { return get_noc_multicast_addr(mx0, my0, mx1, my1, addr); };
    auto row_off = [](uint32_t i) { return (i < 16 ? 0u : 2048u) + (i % 16) * 64; };
    constexpr uint32_t USED = C * T;

    cb_reserve_back(cb_tot, K1);
    const uint32_t tot = get_write_ptr(cb_tot);
    uint32_t tot2 = 0;
    if constexpr (NORM) {
        cb_reserve_back(cb_tot2, NQ);
        tot2 = get_write_ptr(cb_tot2);
    }
    if (root) {
        cb_reserve_back(cb_gath, 2 * K1 * NSEL);
        if constexpr (NORM) {
            cb_reserve_back(cb_gath2, 2 * NQ * NSEL);
        }
    }
    const uint32_t g1 = get_write_ptr(cb_gath);
    const uint32_t g2 = NORM ? get_write_ptr(cb_gath2) : 0;
    noc_semaphore_inc(get_noc_addr(rx, ry, s_ready), 1);
    if (root) {
        noc_semaphore_wait(sem(s_ready), C);
        *sem(s_go) = 1;
        noc_semaphore_set_multicast(s_go, mc(s_go), C - 1);
    }
    noc_semaphore_wait(sem(s_go), 1);
    if (root) {  // rows past C T of each part's last gather tile -> 0 (the 0 / 1 matrices skip them; 0 x NaN would not)
        constexpr uint32_t r0 = USED % 32;
        if constexpr (r0 != 0) {
            cb_reserve_back(cb_zero, 1);
            const uint32_t z = get_write_ptr(cb_zero);
            noc_async_read(zz.get_noc_addr(0), z, 2048);
            noc_async_read_barrier();
            constexpr uint32_t P1 = 2 * K1, P2 = NORM ? 2 * NQ : 0;
            for (uint32_t p = 0; p < P1 + P2; ++p) {
                const uint32_t tile = (p < P1 ? g1 + p * NSEL * F32 : g2 + (p - P1) * NSEL * F32) + (NSEL - 1) * F32;
                if (r0 < 16) {
                    const uint32_t nb = (16 - r0) * 64;
                    noc_async_write(z, get_noc_addr(tile + r0 * 64), nb);
                    noc_async_write(z, get_noc_addr(tile + 1024 + r0 * 64), nb);
                    noc_async_write(z, get_noc_addr(tile + 2048), 2048);
                } else {
                    const uint32_t nb = (32 - r0) * 64;
                    noc_async_write(z, get_noc_addr(tile + 2048 + (r0 - 16) * 64), nb);
                    noc_async_write(z, get_noc_addr(tile + 3072 + (r0 - 16) * 64), nb);
                }
            }
            noc_async_write_barrier();
        }
    }

    if constexpr (TAIL) {
        const auto x_o = TensorAccessor(x_args, ca(3), 2048);
        for (uint32_t i = 0; i < PER; ++i) {
            const uint32_t t = t0 + i;
            cb_wait_front(cb_so, 3);
            const uint32_t so = get_read_ptr(cb_so);
            for (uint32_t j = 0; j < 3; ++j) {
                const auto as = TensorAccessor(s_args, ca(j), F32);
                write_rows<T, 4>(so + j * F32, as.get_noc_addr(t));
            }
            cb_wait_front(cb_xo, 1);
            write_rows<T, 2>(get_read_ptr(cb_xo), x_o.get_noc_addr(t));
            noc_async_write_barrier();
            cb_pop_front(cb_so, 3);
            cb_pop_front(cb_xo, 1);
        }
    }

    auto round = [&](uint32_t cb_p, uint32_t parts, uint32_t cb_g, uint32_t g, uint32_t cb_t, uint32_t tb,
                     uint32_t done_val) {
        cb_wait_front(cb_p, parts);
        const uint32_t src = get_read_ptr(cb_p);
        for (uint32_t p = 0; p < parts; ++p) {
            for (uint32_t t = 0; t < T; ++t) {
                const uint32_t rho = c * T + t;
                const uint32_t d = g + (p * NSEL + rho / 32) * F32 + row_off(rho % 32);
                const uint32_t s = src + p * F32 + row_off(t);
                noc_async_write(s, get_noc_addr(rx, ry, d), 64);
                noc_async_write(s + 1024, get_noc_addr(rx, ry, d + 1024), 64);
            }
        }
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(rx, ry, s_g), 1);
        cb_pop_front(cb_p, parts);
        const uint32_t ntot = parts / 2;
        if (root) {
            noc_semaphore_wait(sem(s_g), C * done_val);
            cb_push_back(cb_g, parts * NSEL);
            cb_wait_front(cb_tout, ntot);
            const uint32_t ts = get_read_ptr(cb_tout);
            for (uint32_t k = 0; k < ntot; ++k) {
                for (uint32_t t = 0; t < T; ++t) {
                    const uint32_t o = k * F32 + row_off(t);
                    noc_async_write_multicast_loopback_src(ts + o, mc(tb + o), 64, C);
                    noc_async_write_multicast_loopback_src(ts + o + 1024, mc(tb + o + 1024), 64, C);
                }
            }
            noc_async_write_barrier();
            cb_pop_front(cb_tout, ntot);
            noc_semaphore_inc_multicast(mc(s_done), 1, C - 1);
            noc_semaphore_inc(get_noc_addr(s_done), 1);
        }
        noc_semaphore_wait(sem(s_done), done_val);
        cb_push_back(cb_t, ntot);
    };
    round(cb_part, 2 * K1, cb_gath, g1, cb_tot, tot, 1);

    if constexpr (NORM) {
        round(cb_part2, 2 * NQ, cb_gath2, g2, cb_tot2, tot2, 2);
    } else {
        for (uint32_t q = 0; q < NQ; ++q) {
            const auto ao = TensorAccessor(x_args, ca(5 + q), 1024 * OE);
            for (uint32_t i = 0; i < PER; ++i) {
                cb_wait_front(cb_out, 1);
                write_rows<T, OE>(get_read_ptr(cb_out), ao.get_noc_addr(t0 + i));
                noc_async_write_barrier();
                cb_pop_front(cb_out, 1);
            }
        }
    }
    if constexpr (BOUND) {
        const auto xn = TensorAccessor(x_args, ca(4), F32);
        for (uint32_t i = 0; i < PER; ++i) {
            cb_wait_front(cb_xn, 1);
            write_rows<T, 4>(get_read_ptr(cb_xn), xn.get_noc_addr(t0 + i));
            noc_async_write_barrier();
            cb_pop_front(cb_xn, 1);
        }
    }
    if constexpr (NORM) {
        for (uint32_t q = 0; q < NQ; ++q) {
            const auto ao = TensorAccessor(x_args, ca(5 + q), 1024 * OE);
            for (uint32_t i = 0; i < PER; ++i) {
                cb_wait_front(cb_out, 1);
                write_rows<T, OE>(get_read_ptr(cb_out), ao.get_noc_addr(t0 + i));
                noc_async_write_barrier();
                cb_pop_front(cb_out, 1);
            }
        }
    }
    *sem(s_ready) = 0;
    *sem(s_go) = 0;
    *sem(s_g) = 0;
    *sem(s_done) = 0;
}
