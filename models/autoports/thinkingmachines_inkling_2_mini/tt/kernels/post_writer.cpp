// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fused decode post-attention section (tt/post.py), writer: the conv's new states, the two cross-core rounds and the
// outputs. A round, one root per row (core t for row t < T): every core writes row t of its P partial tiles (hi / lo
// halves) to row c of root t's gather tiles and increments root t's semaphore; each root's compute reduces them (0 / 1
// matrices) into its totals (row 0), which root t writes into row t of core 0's totals buffer; core 0 multicasts the T
// rows (one multicast per face: concurrent multicasts from several roots serialize) to every core's, then increments
// every core's done semaphore. Start: every core tells core 0 it is running; its go releases the writes.
//
// common args: s0, s1, s2, resid, out, mcast x0, y0, x1, y1, zeros, logits, root t x, y (t < T)

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "red.hpp"

template <uint32_t T, uint32_t E>
inline void write_rows(uint32_t src, uint64_t dst) {
    constexpr uint32_t FACE = 256 * E;  // (rows 0 .. T-1 of a face are contiguous: one write per face)
    noc_async_write(src, dst, T * 16 * E);
    noc_async_write(src + FACE, dst + FACE, T * 16 * E);
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
    const auto zz = TensorAccessor(z_args, ca(9), 2048);
    auto rxy = [&](uint32_t t, uint32_t a) { return get_noc_addr(ca(11 + 2 * t), ca(12 + 2 * t), a); };
    const bool root = c < T;  // root of row c

    const uint32_t s_ready = get_semaphore(0), s_go = get_semaphore(1), s_g = get_semaphore(2);
    const uint32_t s_done = get_semaphore(3), s_t = get_semaphore(4);
    auto sem = [](uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a); };
    auto mc = [&](uint32_t addr) { return get_noc_multicast_addr(mx0, my0, mx1, my1, addr); };
    auto row_off = [](uint32_t i) { return (i < 16 ? 0u : 2048u) + (i % 16) * 64; };

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
        noc_semaphore_inc(rxy(0, s_ready), 1);
        if (c == 0) {
            noc_semaphore_wait(sem(s_ready), C);
            *sem(s_go) = 1;
            noc_semaphore_set_multicast(s_go, mc(s_go), C - 1);
        }
        noc_semaphore_wait(sem(s_go), 1);
    }
    if (root) {
        // zero the unused gather rows (red.hpp): a zero block (2 KB, read once) copied over them
        {
            cb_reserve_back(cb_zero, 1);
            const uint32_t z = get_write_ptr(cb_zero);
            noc_async_read(zz.get_noc_addr(0), z, 2048);
            noc_async_read_barrier();
#if POST_RT
            constexpr uint32_t P3 = 2 * POST_RT;
            const uint32_t g3 = get_write_ptr(POST_CB_G3);
#else
            constexpr uint32_t P3 = 0;
            const uint32_t g3 = 0;
#endif
            for (uint32_t p = 0; p < 4 + (NORM ? 2 : 0) + P3; ++p) {
                red_zero<POST_RK, C, NQ>(z, p < 4 ? g1 + p * NQ * F32 : p < 4 + (NORM ? 2 : 0) ? g2 + (p - 4) * NQ * F32
                                                                    : g3 + (p - 4 - (NORM ? 2 : 0)) * NQ * F32);
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

    // parts' row t -> row c of root t's gather (rho = c), then root t's semaphore
    auto send = [&](uint32_t src, uint32_t parts, uint32_t g) {
        const uint32_t d = g + red_off<POST_RK>(c);
        for (uint32_t t = 0; t < T; ++t) {
            for (uint32_t p = 0; p < parts; ++p) {
                const uint32_t s = src + p * F32 + row_off(t);
                noc_async_write(s, rxy(t, d + p * NQ * F32), 64);
                noc_async_write(s + 1024, rxy(t, d + p * NQ * F32 + 1024), 64);
            }
        }
        noc_async_write_barrier();
        for (uint32_t t = 0; t < T; ++t) {
            noc_semaphore_inc(rxy(t, s_g), 1);
        }
    };
    // one round: parts -> the roots, root t's totals (ntot tiles, row 0) -> row t of everyone's
    auto round = [&](uint32_t cb_p, uint32_t parts, uint32_t cb_g, uint32_t g, uint32_t cb_t, uint32_t tb,
                     uint32_t ntot, uint32_t done_val) {
        cb_wait_front(cb_p, parts);
        send(get_read_ptr(cb_p), parts, g);
        cb_pop_front(cb_p, parts);
        if (root) {
            noc_semaphore_wait(sem(s_g), C * done_val);
            cb_push_back(cb_g, parts * NQ);
            cb_wait_front(cb_tout, ntot);
            const uint32_t ts = get_read_ptr(cb_tout);
            if constexpr (T == 1) {
                for (uint32_t k = 0; k < ntot; ++k) {
                    noc_async_write_multicast_loopback_src(ts + k * F32, mc(tb + k * F32), 64, C);
                    noc_async_write_multicast_loopback_src(ts + k * F32 + 1024, mc(tb + k * F32 + 1024), 64, C);
                }
            } else {  // row 0 -> row c of core 0's totals
                for (uint32_t k = 0; k < ntot; ++k) {
                    const uint32_t o = tb + k * F32 + row_off(c);
                    noc_async_write(ts + k * F32, rxy(0, o), 64);
                    noc_async_write(ts + k * F32 + 1024, rxy(0, o + 1024), 64);
                }
            }
            noc_async_write_barrier();
            cb_pop_front(cb_tout, ntot);
            if constexpr (T > 1) {
                if (c != 0) {
                    noc_semaphore_inc(rxy(0, s_t), 1);
                } else {  // rows 0..T-1 (contiguous per face) -> everyone
                    noc_semaphore_wait(sem(s_t), (T - 1) * done_val);
                    for (uint32_t k = 0; k < ntot; ++k) {
                        const uint32_t o = tb + k * F32;
                        noc_async_write_multicast(o, mc(o), T * 64, C - 1);
                        noc_async_write_multicast(o + 1024, mc(o + 1024), T * 64, C - 1);
                    }
                    noc_async_write_barrier();
                }
            }
            if (c == 0) {
                noc_semaphore_inc_multicast(mc(s_done), 1, C - 1);
                noc_semaphore_inc(get_noc_addr(s_done), 1);
            }
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
#if POST_DUP
        write_rows<T, OUTB / 1024>(get_read_ptr(cb_out), oo.get_noc_addr(t0 + i) + 16 * (OUTB / 1024));
#endif
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
#if POST_RT
    {  // router round: the partial logits' row t -> root t's gather; root t writes logits row t
        constexpr uint32_t ET = POST_RT, parts = 2 * ET;
        constexpr auto l_args = TensorAccessorArgs<z_args.next_compile_time_args_offset()>();
        cb_wait_front(POST_CB_P3, parts);
        send(get_read_ptr(POST_CB_P3), parts, get_write_ptr(POST_CB_G3));
        cb_pop_front(POST_CB_P3, parts);
        if (root) {
            cb_reserve_back(POST_CB_G3, parts * NQ);
            noc_semaphore_wait(sem(s_g), C * (NORM ? 3 : 2));
            cb_push_back(POST_CB_G3, parts * NQ);
            const auto al = TensorAccessor(l_args, ca(10), F32);
            cb_wait_front(POST_CB_T3, ET);
            const uint32_t ts = get_read_ptr(POST_CB_T3);
            for (uint32_t e = 0; e < ET; ++e) {
                const uint64_t dl = al.get_noc_addr(e) + c * 64;
                write_rows<1, 4>(ts + e * F32, dl);
#if POST_DUP
                write_rows<1, 4>(ts + e * F32, dl + 64);
#endif
            }
            noc_async_write_barrier();
            cb_pop_front(POST_CB_T3, ET);
        }
    }
#endif
    *sem(s_ready) = 0;
    *sem(s_go) = 0;
    *sem(s_g) = 0;
    *sem(s_done) = 0;
    *sem(s_t) = 0;
}
