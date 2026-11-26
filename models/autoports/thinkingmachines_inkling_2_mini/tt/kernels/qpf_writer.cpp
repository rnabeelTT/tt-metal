// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Buffer-fed streamed decode projection, activation in + output out (BRISC): kernels/mstream_writer.cpp with the output
// CB / tile size per segment (kind 0: cb_oq, FP32 q or BF16 o, kind 1: cb_o BF16). Output column i of a segment goes to page
// col0 + 8 i (an interleaved weight's bank columns, n = bank + 8 c).
//
// ARP_T (one kind-0 BF16 segment: the o_proj; tt/arpush.py): the output goes into the fused all-reduce instead
// (tt/ccl.py push): each column's ARP_T live rows into this die's slot on the shard cores and the link senders' payload,
// packed as arpack.cpp (tile n's chunk at 64 T n: face 0's T rows, then face 1's), then one increment per touched shard
// core / sender.
//
// runtime args: [nseg, then per segment: x_addr, out_addr, col0, ncols, kind; padded to 16, then nq (q blocks)]
// (ARP_T: + slot, K, shard noc (x, y) x K, sender noc (x, y) x links)

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "qpf_rcb.hpp"

void kernel_main() {
    const uint32_t nseg = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_oq = get_compile_time_arg_val(1);
    constexpr uint32_t cb_o = get_compile_time_arg_val(2);
    constexpr uint32_t kt = get_compile_time_arg_val(3);
    constexpr uint32_t rows = get_compile_time_arg_val(4);
    constexpr uint32_t H = get_compile_time_arg_val(5);
    constexpr auto x_args = TensorAccessorArgs<6>();
    constexpr auto o_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    // ack_w: this core's reader is on NOC1, so the GCB credits go back from here (NOC0) once compute is done
    constexpr uint32_t ack_w = get_compile_time_arg_val(o_args.next_compile_time_args_offset());
    constexpr uint32_t rcb = get_compile_time_arg_val(o_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t drain = get_compile_time_arg_val(o_args.next_compile_time_args_offset() + 2);
    // oqb: a kind-0 output tile's bytes (FP32 q: 4096; BF16 o_proj: 2048)
    constexpr uint32_t oqb = get_compile_time_arg_val(o_args.next_compile_time_args_offset() + 3);
    const uint32_t nq = get_arg_val<uint32_t>(16);
    if constexpr (ack_w && drain) {
        for (uint32_t i = 0; i < nq; ++i) {
            while (!remote_avail(rcb, i + 1)) {
            }
        }
        if (nq) {
            experimental::remote_cb_pop_front(rcb, nq);
            experimental::update_remote_cb_config_in_l1(rcb);
            noc_async_atomic_barrier();
        }
        return;
    }
    static_assert(rows <= 32, "one tile row");
    constexpr uint32_t row_bytes = rows * 32;

    if (nseg == 0) {
        return;
    }
    constexpr uint32_t XT = H * 64;
    cb_reserve_back(cb_x, kt * nseg);
    for (uint32_t s = 0; s < nseg; ++s) {
        const uint32_t xl1 = get_write_ptr(cb_x) + s * kt * XT;
        const auto s_x = TensorAccessor(x_args, get_arg_val<uint32_t>(1 + 5 * s), 2048);
        for (uint32_t k = 0; k < kt; ++k) {
            const uint64_t src = s_x.get_noc_addr(k);
            if constexpr (rows > 16) {
                noc_async_read(src, xl1 + k * XT, 2048);
            } else {
                noc_async_read(src, xl1 + k * XT, row_bytes);
                noc_async_read(src + 512, xl1 + k * XT + XT / 2, row_bytes);
            }
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, kt * nseg);

#ifdef ARP_T
    static_assert(H < 32 && ARP_T <= rows, "tiny output tiles");
    constexpr uint32_t PA = 17;
    const uint32_t slot = get_arg_val<uint32_t>(PA);
    const uint32_t K = get_arg_val<uint32_t>(PA + 1);
    auto shard_noc = [&](uint32_t k, uint32_t addr) {
        return get_noc_addr(get_arg_val<uint32_t>(PA + 2 + 2 * k), get_arg_val<uint32_t>(PA + 3 + 2 * k), addr);
    };
    auto send_noc = [&](uint32_t l, uint32_t addr) {
        return get_noc_addr(
            get_arg_val<uint32_t>(PA + 2 + 2 * K + 2 * l), get_arg_val<uint32_t>(PA + 3 + 2 * K + 2 * l), addr);
    };
    const uint32_t tx = get_write_ptr(ARP_CB_TX);
    auto put = [&](uint32_t s, uint32_t b, uint32_t n) {  // (a run may cross a shard tile)
        while (n > 0) {
            const uint32_t k = b / 2048, o = b % 2048, m = 2048 - o < n ? 2048 - o : n;
            noc_async_write(s, shard_noc(k, slot + o), m);
            noc_async_write(s, send_noc(k / ARP_KPL, tx + (k % ARP_KPL) * 2048 + o), m);
            b += m;
            s += m;
            n -= m;
        }
    };
    {
        const uint32_t col0 = get_arg_val<uint32_t>(3);
        const uint32_t ncols = get_arg_val<uint32_t>(4);
        for (uint32_t i = 0; i < ncols; ++i) {
            cb_wait_front(cb_oq, 1);
            const uint32_t n = col0 + i * 8;
            put(get_read_ptr(cb_oq), 64 * ARP_T * n, 32 * ARP_T);
            put(get_read_ptr(cb_oq) + H * 32, 64 * ARP_T * n + 32 * ARP_T, 32 * ARP_T);
            noc_async_writes_flushed();
            cb_pop_front(cb_oq, 1);
        }
        noc_async_write_barrier();
        uint32_t lk = 0xffffffff, ll = 0xffffffff;
        for (uint32_t i = 0; i < ncols; ++i) {  // (ascending n: the touched shard tiles / links ascend)
            const uint32_t b = 64 * ARP_T * (col0 + i * 8);
            for (uint32_t k = b / 2048; k <= (b + 64 * ARP_T - 1) / 2048; ++k) {
                if (k != lk) {
                    noc_semaphore_inc(shard_noc(k, get_semaphore(ARP_SEM_SH)), 1);
                    lk = k;
                }
                if (k / ARP_KPL != ll) {
                    ll = k / ARP_KPL;
                    noc_semaphore_inc(send_noc(ll, get_semaphore(ARP_SEM_TX)), 1);
                }
            }
        }
        noc_async_atomic_barrier();
    }
#else
    for (uint32_t s = 0; s < nseg; ++s) {
        const uint32_t kind = get_arg_val<uint32_t>(5 + 5 * s);
        const uint32_t OB = kind == 0 ? oqb : 2048;
        const uint32_t cb_out = kind == 0 ? cb_oq : cb_o;
        const auto s_o = TensorAccessor(o_args, get_arg_val<uint32_t>(2 + 5 * s), OB);
        const uint32_t col0 = get_arg_val<uint32_t>(3 + 5 * s);
        const uint32_t ncols = get_arg_val<uint32_t>(4 + 5 * s);
        const uint32_t E = OB / 1024;
        for (uint32_t i = 0; i < ncols; ++i) {
            cb_wait_front(cb_out, 1);
            const uint64_t dst = s_o.get_noc_addr(col0 + i * 8);
            if constexpr (H == 32) {
                noc_async_write(get_read_ptr(cb_out), dst, OB);
            } else {
                noc_async_write(get_read_ptr(cb_out), dst, rows * 16 * E);
                noc_async_write(get_read_ptr(cb_out) + H * 16 * E, dst + 256 * E, rows * 16 * E);
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_out, 1);
        }
    }
    noc_async_write_barrier();
#endif
    if constexpr (ack_w) {
        if (nq) {  // (every output written: compute has popped every q block)
            experimental::remote_cb_pop_front(rcb, nq);
            experimental::update_remote_cb_config_in_l1(rcb);
            noc_async_atomic_barrier();
        }
    }
}
