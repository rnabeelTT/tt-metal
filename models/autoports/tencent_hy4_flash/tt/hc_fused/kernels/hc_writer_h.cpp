// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fused hyper-connection, writer, HALF-TILE decode variant (hc_compute_h.cpp): half-tile slots (rows 0..15); X' half
// tiles -> cb_xf full tiles (rows 16..31 zeroed: the compute's in1 of X' X'^T, and the residual pages written, so the
// padding rows 16..31 stay zero); in / w written as their half + zeros.
// (Full-tile notes follow.) Fused hyper-connection, writer (RISCV_0). Per row tile r of the group (r = g, g + G, ... < TR; decode: one):
// PRE: this core's two partial tiles (mix, sum of squares) into slots 2c, 2c + 1 of cb_slots on every core of its
// group (the addnorm exchange: a ready handshake once this core's slots are free, the writes, one semaphore increment
// per core), waits for all NC cores' partials and hands the slots to the compute. Then writes (POST) the new residual
// tiles X', (PRE) the sublayer input tiles and (core 0 of the group) the weight tile.
//
// runtime args: [xn, in, w, c, g, noc x/y of the group's NC cores ...]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

#ifndef HR
#define HR 16  // rows of the partial tiles (16: faces 0 / 1 whole; 8: their rows 0..7)
#endif

void kernel_main() {
    const uint32_t xn_addr = get_arg_val<uint32_t>(0);
    const uint32_t in_addr = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);
    const uint32_t c = get_arg_val<uint32_t>(3);

    constexpr uint32_t J = get_compile_time_arg_val(0);
    constexpr uint32_t HT = get_compile_time_arg_val(1);
    constexpr uint32_t POST = get_compile_time_arg_val(2);
    constexpr uint32_t PRE = get_compile_time_arg_val(3);
    constexpr uint32_t NC = get_compile_time_arg_val(4);
    constexpr uint32_t SEM = get_compile_time_arg_val(5);
    constexpr uint32_t SEM_READY = get_compile_time_arg_val(6);
    constexpr uint32_t cb_p = get_compile_time_arg_val(7);
    constexpr uint32_t cb_slots = get_compile_time_arg_val(8);
    constexpr uint32_t cb_xn = get_compile_time_arg_val(9);
    constexpr uint32_t cb_in = get_compile_time_arg_val(10);
    constexpr uint32_t cb_w = get_compile_time_arg_val(11);
    constexpr uint32_t TR = get_compile_time_arg_val(12);
    constexpr uint32_t G = get_compile_time_arg_val(13);
    constexpr auto xn_args = TensorAccessorArgs<14>();
    constexpr auto in_args = TensorAccessorArgs<xn_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr uint32_t BF = 2048, F32 = 4096, BFH = HR * 64, F32H = HR * 128;
    constexpr uint32_t cb_xf = 25;
    const uint32_t zeros = MEM_ZEROS_BASE;
    // n zero bytes to (local or remote) NOC address dst
    auto zero_to = [&](uint64_t dst, uint32_t n) {
        while (n) {
            const uint32_t k = n < MEM_ZEROS_SIZE ? n : MEM_ZEROS_SIZE;
            noc_async_write(zeros, dst, k);
            dst += k;
            n -= k;
        }
    };
    // a partial HR x 32 tile at local src -> the full tile at NOC address dst (element bytes eb), every other row zero
    auto write_full = [&](uint32_t src, uint64_t dst, uint32_t eb) {
        const uint32_t part = HR * 16 * eb, face = 256 * eb;
        noc_async_write(src, dst, part);
        if (part < face) {
            zero_to(dst + part, face - part);
        }
        noc_async_write(src + part, dst + face, part);
        if (part < face) {
            zero_to(dst + face + part, face - part);
        }
        zero_to(dst + 2 * face, 2 * face);
    };

    const uint32_t g = get_arg_val<uint32_t>(4);
    constexpr uint32_t NOC0 = 5;  // first noc coordinate arg
    const auto s_xn = TensorAccessor(xn_args, xn_addr, BF);
    const auto s_w = TensorAccessor(w_args, w_addr, F32);
    const auto s_in = TensorAccessor(in_args, in_addr, BF);
    uint32_t xf_base = 0;
    for (uint32_t r = g; r < TR; r += G) {
        if constexpr (POST) {
            // X' half tiles -> full tiles (rows 16..31 zero) in cb_xf, for the compute (X' X'^T) and the residual write
            cb_wait_front(cb_xn, 4 * J);
            cb_reserve_back(cb_xf, 4 * J);
            const uint32_t src = get_read_ptr(cb_xn), dst = get_write_ptr(cb_xf);
            xf_base = dst;
            for (uint32_t t = 0; t < 4 * J; ++t) {
                write_full(src + t * BFH, get_noc_addr(dst + t * BF), 2);
            }
            noc_async_write_barrier();
            cb_push_back(cb_xf, 4 * J);
        }
        if constexpr (PRE) {
            DeviceZoneScopedN("HCW_XCH");
            const uint32_t sem = get_semaphore(SEM);
            const uint32_t ready = get_semaphore(SEM_READY);
            // this core's slots free (the compute popped the previous row tile's) before the others may write them
            cb_reserve_back(cb_slots, 2 * NC);
            for (uint32_t k = 0; k < NC; ++k) {
                const uint32_t x = get_arg_val<uint32_t>(NOC0 + 2 * k), y = get_arg_val<uint32_t>(NOC0 + 1 + 2 * k);
                noc_semaphore_inc(get_noc_addr(x, y, ready), 1);
            }
            const uint32_t slots = get_write_ptr(cb_slots);  // the same address on every core of the program
            cb_wait_front(cb_p, 2);
            volatile tt_l1_ptr uint32_t* ready_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready);
            noc_semaphore_wait(ready_ptr, NC);
            noc_semaphore_set(ready_ptr, 0);
            const uint32_t p = get_read_ptr(cb_p);
            for (uint32_t k = 0; k < NC; ++k) {
                const uint32_t x = get_arg_val<uint32_t>(NOC0 + 2 * k), y = get_arg_val<uint32_t>(NOC0 + 1 + 2 * k);
                noc_async_write(p, get_noc_addr(x, y, slots + 2 * c * F32H), 2 * F32H);
            }
            noc_async_write_barrier();
            for (uint32_t k = 0; k < NC; ++k) {
                const uint32_t x = get_arg_val<uint32_t>(NOC0 + 2 * k), y = get_arg_val<uint32_t>(NOC0 + 1 + 2 * k);
                noc_semaphore_inc(get_noc_addr(x, y, sem), 1);
            }
            cb_pop_front(cb_p, 2);
            volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem);
            noc_semaphore_wait(sem_ptr, NC);
            noc_semaphore_set(sem_ptr, 0);
            cb_push_back(cb_slots, 2 * NC);
        }
        if constexpr (POST) {
            uint32_t l1 = xf_base;  // (pushed above; the compute only reads it)
            for (uint32_t s = 0; s < 4; ++s) {
                for (uint32_t jj = 0; jj < J; ++jj) {
                    noc_async_write(l1, s_xn.get_noc_addr((s * TR + r) * HT + c * J + jj), BF);
                    l1 += BF;
                }
            }
        }
        if constexpr (PRE) {
            cb_wait_front(cb_w, 1);  // every core's compute packs it; core 0 writes it, every core pops it
            if (c == 0) {
                const uint64_t d = s_w.get_noc_addr(r);
                write_full(get_read_ptr(cb_w), d, 4);
            }
            cb_wait_front(cb_in, J);
            uint32_t l1 = get_read_ptr(cb_in);
            for (uint32_t jj = 0; jj < J; ++jj) {
                const uint64_t d = s_in.get_noc_addr(r * HT + c * J + jj);
                write_full(l1, d, 2);
                l1 += BFH;
            }
        }
        noc_async_write_barrier();
        noc_async_atomic_barrier();
        if constexpr (POST) {
            cb_pop_front(cb_xn, 4 * J);
        }
        if constexpr (PRE) {
            cb_pop_front(cb_w, 1);
            cb_pop_front(cb_in, J);
        }
    }
}
