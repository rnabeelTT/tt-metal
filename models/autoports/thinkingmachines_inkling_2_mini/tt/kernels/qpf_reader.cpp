// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Buffer-fed streamed decode projection, weight reader (NCRISC).
// 1. The prefetched weight: the DRISC tensor prefetcher pushes this receiver's K-blocks (one remote page = kbq k rows x
//    gq tiles, k-major) into the global CB rcb; the local CB cb_wq aliases the same L1 (its pointers aligned to the
//    remote read pointer at kernel start). Each landed page is published to compute as one local block; a page's remote
//    credit goes back once compute has popped it. With a ring shorter than the tensor the publishing waits on those
//    credits (pops interleave); otherwise every page is published at once and the credits return at the end.
// 2. Then any DRAM segments: kernels/ilstream_reader.cpp's interleaved stream into cb_w (blocks of kb x g).
//
// runtime args: [nq (q blocks, 0: none), bank, vc, nseg, then per segment: w_addr, col0, ncols]
// compile args: cb_wq, cb_w, kt, tile_bytes, nb, nblk, kb, g, qblk (tiles per q block), rcb, drain, ack_w

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/remote_circular_buffer.h"
#include "qpf_rcb.hpp"

void kernel_main() {
    constexpr uint32_t cb_wq = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t kt = get_compile_time_arg_val(2);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t nb = get_compile_time_arg_val(4);
    constexpr uint32_t nblk = get_compile_time_arg_val(5);
    constexpr uint32_t kb = get_compile_time_arg_val(6);
    constexpr uint32_t g = get_compile_time_arg_val(7);
    constexpr uint32_t qblk = get_compile_time_arg_val(8);
    constexpr uint32_t rcb = get_compile_time_arg_val(9);
    constexpr bool drain = get_compile_time_arg_val(10);  // (pop-only: resync of an orphan request, stream-rate tests)
    // ack_w: the GCB credits go back from the writer (BRISC, NOC0): this reader reads DRAM on NOC1 (the GCB's sender
    // coordinates are NOC0's: a NOC1 credit atomic never acks)
    constexpr bool ack_w = get_compile_time_arg_val(11);
    constexpr uint32_t blk = kb * g;
    constexpr uint32_t depth = nblk - 1;
    constexpr uint32_t ring_end = nblk * blk * tile_bytes;

    const uint32_t nq = get_arg_val<uint32_t>(0);
    const uint32_t bank = get_arg_val<uint32_t>(1);
    const uint32_t vc = get_arg_val<uint32_t>(2);
    const uint32_t nseg = get_arg_val<uint32_t>(3);

    if constexpr (drain) {
        if constexpr (ack_w) {
            return;
        }
        for (uint32_t i = 0; i < nq; ++i) {
            while (!remote_avail(rcb, 1)) {
            }
            experimental::remote_cb_pop_front(rcb, 1);
        }
        experimental::update_remote_cb_config_in_l1(rcb);
        noc_async_atomic_barrier();
        return;
    }
    uint32_t pushed = 0, popped = 0;
    uint32_t fifo_tiles = 0;
    if (nq) {
        fifo_tiles = get_local_cb_interface(cb_wq).fifo_num_pages;
        while (pushed < nq) {
            // (landed pages count from the oldest un-acked one: those published and not yet credited, plus the next)
            if (remote_avail(rcb, pushed - popped + 1)) {
                cb_reserve_back(cb_wq, qblk);
                cb_push_back(cb_wq, qblk);
                ++pushed;
            } else if (!ack_w && popped < pushed && cb_pages_reservable_at_back(cb_wq, fifo_tiles - (pushed - popped - 1) * qblk)) {
                experimental::remote_cb_pop_front(rcb, 1);  // compute is done with block `popped`
                ++popped;
            }
        }
    }

    if (nseg) {
        const uint64_t bank_noc = get_noc_addr_from_bank_id<true>(bank, get_arg_val<uint32_t>(4));
        noc_async_read_one_packet_set_state<true>(bank_noc, tile_bytes, vc);
        reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
        const uint32_t cb_base = get_write_ptr(cb_w);
        uint32_t l1 = cb_base;
        uint32_t inflight = 0;
        uint32_t trid = 1;
        uint32_t trid_to_wait = 1;
        for (uint32_t s = 0; s < nseg; ++s) {
            const uint32_t base = (uint32_t)get_noc_addr_from_bank_id<true>(bank, get_arg_val<uint32_t>(4 + 3 * s));
            const uint32_t col0 = get_arg_val<uint32_t>(5 + 3 * s);
            const uint32_t ncols = get_arg_val<uint32_t>(6 + 3 * s);
            for (uint32_t c = 0; c < ncols; c += g) {
                const uint32_t gw = ncols - c < g ? ncols - c : g;
                for (uint32_t k0 = 0; k0 < kt; k0 += kb) {
                    cb_reserve_back(cb_w, blk * (inflight + 1));
                    noc_async_read_set_trid(trid);
                    for (uint32_t j = 0; j < gw; ++j) {
                        uint32_t src = (k0 * nb + col0 + c + j) * tile_bytes;
                        uint32_t dst = l1 + j * tile_bytes;
                        for (uint32_t k = 0; k < kb; ++k) {
                            noc_async_read_one_packet_with_state_with_trid(base, src, dst, trid);
                            src += nb * tile_bytes;
                            dst += g * tile_bytes;
                        }
                    }
                    l1 += blk * tile_bytes;
                    if (l1 >= cb_base + ring_end) {
                        l1 = cb_base;
                    }
                    trid = trid == depth + 1 ? 1 : trid + 1;
                    if (++inflight == depth) {
                        noc_async_read_barrier_with_trid(trid_to_wait);
                        cb_push_back(cb_w, blk);
                        trid_to_wait = trid_to_wait == depth + 1 ? 1 : trid_to_wait + 1;
                        --inflight;
                    }
                }
            }
        }
        for (; inflight > 0; --inflight) {
            noc_async_read_barrier_with_trid(trid_to_wait);
            cb_push_back(cb_w, blk);
            trid_to_wait = trid_to_wait == depth + 1 ? 1 : trid_to_wait + 1;
        }
        noc_async_read_one_packet_set_state<true>(get_noc_addr(MEM_ZEROS_BASE), 0, 1);
        noc_async_read_set_trid(0);
    }

    if (nq && !ack_w) {
        for (; popped < pushed; ++popped) {
            while (!cb_pages_reservable_at_back(cb_wq, fifo_tiles - (pushed - popped - 1) * qblk)) {
                invalidate_l1_cache();
            }
            experimental::remote_cb_pop_front(rcb, 1);
        }
        experimental::update_remote_cb_config_in_l1(rcb);
        noc_async_atomic_barrier();
    }
}
