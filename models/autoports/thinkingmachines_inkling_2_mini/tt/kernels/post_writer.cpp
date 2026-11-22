// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fused decode post-attention section (tt/post.py), writer: the conv's new states, the two cross-core rounds and the
// outputs. A round: every core writes the live rows (t < T) of its P partial tiles (hi / lo halves) to row c T + t of
// the root's gather tiles and increments the root's semaphore; the root's compute reduces them (0 / 1 matrices) into
// totals, which the root multicasts (rows t < T) into every core's totals buffer, then increments every core's done
// semaphore. Start: every core tells the root it is running; the root's go releases the writes into it.
//
// runtime args: t0, c, s0, s1, s2, resid, out, mcast x0, y0, x1, y1, root x, root y

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

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
    constexpr uint32_t F32 = 4096;
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
    const uint32_t rx = ca(9), ry = ca(10);
    const auto zz = TensorAccessor(z_args, ca(11), 2048);
    const bool root = c == 0;

    const uint32_t s_ready = get_semaphore(0), s_go = get_semaphore(1), s_g = get_semaphore(2);
    const uint32_t s_done = get_semaphore(3);
    auto sem = [](uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a); };
    auto mc = [&](uint32_t addr) { return get_noc_multicast_addr(mx0, my0, mx1, my1, addr); };
    auto row_off = [](uint32_t i) { return (i < 16 ? 0u : 2048u) + (i % 16) * 64; };
    constexpr uint32_t USED = C * T;

    // receive buffers (the same addresses on every core)
    cb_reserve_back(cb_tot, 2);
    const uint32_t tot = get_write_ptr(cb_tot);
    uint32_t tot2 = 0;
    if constexpr (NORM) {
        cb_reserve_back(cb_tot2, 1);
        tot2 = get_write_ptr(cb_tot2);
    }
    uint32_t g1 = 0, g2 = 0;
    if (root) {
        cb_reserve_back(cb_gath, 4 * NQ);
        g1 = get_write_ptr(cb_gath);
        if constexpr (NORM) {
            cb_reserve_back(cb_gath2, 2 * NQ);
            g2 = get_write_ptr(cb_gath2);
        }
    } else {
        g1 = get_write_ptr(cb_gath);  // (the root's buffer address: same CB layout)
        if constexpr (NORM) {
            g2 = get_write_ptr(cb_gath2);
        }
    }
    {
        DeviceZoneScopedN("P_W_HS");
        noc_semaphore_inc(get_noc_addr(rx, ry, s_ready), 1);
        if (root) {
            noc_semaphore_wait(sem(s_ready), C);
            *sem(s_go) = 1;
            noc_semaphore_set_multicast(s_go, mc(s_go), C - 1);
        }
        noc_semaphore_wait(sem(s_go), 1);
    }
    if (root) {
        // zero the rows past C T of every part's last tile (the 0 / 1 matrices skip them; 0 x NaN would not): a zero
        // block (2 KB, read once) copied over the contiguous face-row spans
        constexpr uint32_t r0 = USED % 32;
        if constexpr (r0 != 0) {
            cb_reserve_back(cb_zero, 1);
            const uint32_t z = get_write_ptr(cb_zero);
            noc_async_read(zz.get_noc_addr(0), z, 2048);
            noc_async_read_barrier();
            for (uint32_t p = 0; p < 4 + (NORM ? 2 : 0); ++p) {
                const uint32_t tile = (p < 4 ? g1 + p * NQ * F32 : g2 + (p - 4) * NQ * F32) + (NQ - 1) * F32;
                if (r0 < 16) {  // faces 0 / 1 rows r0..15, faces 2 / 3 whole
                    const uint32_t n = (16 - r0) * 64;
                    noc_async_write(z, get_noc_addr(tile + r0 * 64), n);
                    noc_async_write(z, get_noc_addr(tile + 1024 + r0 * 64), n);
                    noc_async_write(z, get_noc_addr(tile + 2048), 2048);
                } else {  // faces 2 / 3 rows r0-16..15
                    const uint32_t n = (32 - r0) * 64;
                    noc_async_write(z, get_noc_addr(tile + 2048 + (r0 - 16) * 64), n);
                    noc_async_write(z, get_noc_addr(tile + 3072 + (r0 - 16) * 64), n);
                }
            }
            noc_async_write_barrier();
        }
    }

    for (uint32_t i = 0; i < PER; ++i) {
        const uint32_t t = t0 + i;
        cb_wait_front(cb_so, 3);
        const uint32_t so = get_read_ptr(cb_so);
        write_rows<T, 4>(so, s0.get_noc_addr(t));  // (live rows only: the other rows are other slots' / padding)
        write_rows<T, 4>(so + F32, s1.get_noc_addr(t));
        write_rows<T, 4>(so + 2 * F32, s2.get_noc_addr(t));
        cb_wait_front(cb_rout, 1);
        write_rows<T, 4>(get_read_ptr(cb_rout), ro.get_noc_addr(t));
        noc_async_write_barrier();
        cb_pop_front(cb_so, 3);
        cb_pop_front(cb_rout, 1);
    }

    // one round: parts -> root, root totals (ntot tiles) -> everyone
    auto round = [&](uint32_t cb_p, uint32_t parts, uint32_t cb_g, uint32_t g, uint32_t cb_t, uint32_t tb,
                     uint32_t ntot, uint32_t done_val) {
        cb_wait_front(cb_p, parts);
        const uint32_t src = get_read_ptr(cb_p);
        for (uint32_t p = 0; p < parts; ++p) {
            for (uint32_t t = 0; t < T; ++t) {
                const uint32_t rho = c * T + t;
                const uint32_t d = g + (p * NQ + rho / 32) * F32 + row_off(rho % 32);
                const uint32_t s = src + p * F32 + row_off(t);
                noc_async_write(s, get_noc_addr(rx, ry, d), 64);
                noc_async_write(s + 1024, get_noc_addr(rx, ry, d + 1024), 64);
            }
        }
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(rx, ry, s_g), 1);
        cb_pop_front(cb_p, parts);
        if (root) {
            noc_semaphore_wait(sem(s_g), C * done_val);
            cb_push_back(cb_g, parts * NQ);
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
    {
        DeviceZoneScopedN("P_W_R1");
        round(cb_part, 4, cb_gath, g1, cb_tot, tot, 2, 1);
    }
    if constexpr (NORM) {
        round(cb_part2, 2, cb_gath2, g2, cb_tot2, tot2, 1, 2);
    }

    for (uint32_t i = 0; i < PER; ++i) {
        cb_wait_front(cb_out, 1);
        write_rows<T, OUTB / 1024>(get_read_ptr(cb_out), oo.get_noc_addr(t0 + i));
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
    *sem(s_ready) = 0;
    *sem(s_go) = 0;
    *sem(s_g) = 0;
    *sem(s_done) = 0;
}
