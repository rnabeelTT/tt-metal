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
//
// common args: out, logits, mcast x0, y0, x1, y1, zeros, root t x, y (t < T)

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
    constexpr uint32_t F32 = 4096;
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
    const bool root = c < T;  // root of row c
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
    if (root) {  // the unused gather rows -> 0 (red.hpp)
        {
            cb_reserve_back(cb_zero, 1);
            const uint32_t z = get_write_ptr(cb_zero);
            noc_async_read(zz.get_noc_addr(0), z, 2048);
            noc_async_read_barrier();
            for (uint32_t p = 0; p < GP; ++p) {
                red_zero<G2_RK, C, NSEL>(z, g + p * NSEL * F32);
            }
            noc_async_write_barrier();
        }
    }

    // one round: parts -> the roots; with a broadcast, the roots' ntot totals -> everyone's tb
    auto round = [&](uint32_t parts, uint32_t cb_t, uint32_t tb, uint32_t k, bool bcast) {
        cb_wait_front(cb_part, parts);
        const uint32_t src = get_read_ptr(cb_part);
        const uint32_t d = g + red_off<G2_RK>(c);
        for (uint32_t t = 0; t < T; ++t) {
            for (uint32_t p = 0; p < parts; ++p) {
                const uint32_t s = src + p * F32 + row_off(t);
                noc_async_write(s, rxy(t, d + p * NSEL * F32), 64);
                noc_async_write(s + 1024, rxy(t, d + p * NSEL * F32 + 1024), 64);
            }
        }
        noc_async_write_barrier();
        for (uint32_t t = 0; t < T; ++t) {
            noc_semaphore_inc(rxy(t, s_g), 1);
        }
        cb_pop_front(cb_part, parts);
        if (root) {
            cb_reserve_back(cb_gath, GT);
            noc_semaphore_wait(sem(s_g), C * k);
            cb_push_back(cb_gath, GT);
        }
        if (!bcast) {
            return;
        }
        const uint32_t ntot = parts / 2;
        if (root) {
            cb_wait_front(cb_tout, ntot);
            const uint32_t ts = get_read_ptr(cb_tout);
            if constexpr (T == 1) {
                for (uint32_t j = 0; j < ntot; ++j) {
                    noc_async_write_multicast_loopback_src(ts + j * F32, mc(tb + j * F32), 64, C);
                    noc_async_write_multicast_loopback_src(ts + j * F32 + 1024, mc(tb + j * F32 + 1024), 64, C);
                }
            } else {  // row 0 -> row c of core 0's totals
                for (uint32_t j = 0; j < ntot; ++j) {
                    const uint32_t o = tb + j * F32 + row_off(c);
                    noc_async_write(ts + j * F32, rxy(0, o), 64);
                    noc_async_write(ts + j * F32 + 1024, rxy(0, o + 1024), 64);
                }
            }
            noc_async_write_barrier();
            cb_pop_front(cb_tout, ntot);
            if constexpr (T > 1) {
                if (c != 0) {
                    noc_semaphore_inc(rxy(0, s_t), 1);
                } else {  // rows 0..T-1 (contiguous per face) -> everyone (one multicaster: concurrent ones serialize)
                    noc_semaphore_wait(sem(s_t), (T - 1) * k);
                    for (uint32_t j = 0; j < ntot; ++j) {
                        const uint32_t o = tb + j * F32;
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
        noc_semaphore_wait(sem(s_done), k);
        cb_push_back(cb_t, ntot);
    };
    {
    DeviceZoneScopedN("G2_W_R1");
    round(4, cb_tot, tot, 1, true);
    }
    {
    DeviceZoneScopedN("G2_W_R2");
    round(4, cb_tot2, tot2, 2, true);
    }
    if constexpr (NORM) {
        DeviceZoneScopedN("G2_W_R3");
        round(2, cb_tot3, tot3, 3, true);
    }
    for (uint32_t i = 0; i < PER; ++i) {
        cb_wait_front(cb_out, 1);
        const uint32_t s = get_read_ptr(cb_out);
        const uint64_t d = ao.get_noc_addr(t0 + i);
        write_rows<T, OE>(s, d);
        if constexpr (DUP) {
            write_rows<T, OE>(s, d + 16 * OE);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
    if constexpr (ROUTER) {
        DeviceZoneScopedN("G2_W_R4");
        round(2 * ET, 0, 0, 4, false);
        if (root) {  // logits row c ([Tp, E] fp32)
            const auto al = TensorAccessor(l_args, ca(1), F32);
            cb_wait_front(cb_tout, ET);
            const uint32_t ts = get_read_ptr(cb_tout);
            for (uint32_t e = 0; e < ET; ++e) {
                const uint64_t d = al.get_noc_addr(e) + c * 64;
                write_rows<1, 4>(ts + e * F32, d);
                if constexpr (DUP) {
                    write_rows<1, 4>(ts + e * F32, d + 64);
                }
            }
            noc_async_write_barrier();
            cb_pop_front(cb_tout, ET);
        }
    }
    *sem(s_ready) = 0;
    *sem(s_go) = 0;
    *sem(s_g) = 0;
    *sem(s_done) = 0;
    *sem(s_t) = 0;
}
