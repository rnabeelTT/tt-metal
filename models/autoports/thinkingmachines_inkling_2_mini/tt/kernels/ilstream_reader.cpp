// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Streamed decode projection, weight reader (NCRISC; tt/mstream.py, interleaved source): the weight is an ordinary
// interleaved [K, N] TILE tensor with N % 8 == 0 tile columns, so tile (k, n) = page k N + n sits in bank n mod 8 at
// bank-local page k (N / 8) + n / 8: bank `bank`'s column c (n = bank + 8 c) is kt tiles at a stride of N / 8 pages.
// This core streams its columns of that bank in groups of up to g columns, one tile per read, kb k rows per pushed
// block of kb x g tiles (k-major in L1; a narrower last group still takes a whole block),
// transaction-id pipelined over a ring of nblk blocks with nblk - 1 in flight.
//
// A core may stream several weights of the same shape back to back (segments: tt/mstream.py groups, e.g. k then v).
//
// runtime args: [bank, vc, nseg, then per segment: w_addr, col0 (column within the bank), ncols]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t bank = get_arg_val<uint32_t>(0);
    const uint32_t vc = get_arg_val<uint32_t>(1);
    const uint32_t nseg = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t kt = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t nb = get_compile_time_arg_val(3);  // N / 8: tile columns per bank
    constexpr uint32_t nblk = get_compile_time_arg_val(4);  // ring blocks
    constexpr uint32_t kb = get_compile_time_arg_val(5);  // k rows per block (divides kt)
    constexpr uint32_t g = get_compile_time_arg_val(6);  // columns per group
    constexpr uint32_t blk = kb * g;  // tiles per block
    constexpr uint32_t depth = nblk - 1;  // blocks in flight
    static_assert(nblk >= 2 && kt % kb == 0 && nblk <= 15, "ring of blocks / transaction ids");
    constexpr uint32_t ring_end = nblk * blk * tile_bytes;

    if (nseg == 0) {
        return;
    }
    const uint64_t bank_noc = get_noc_addr_from_bank_id<true>(bank, get_arg_val<uint32_t>(3));
    noc_async_read_one_packet_set_state<true>(bank_noc, tile_bytes, vc);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    const uint32_t cb_base = get_write_ptr(cb_w);
    uint32_t l1 = cb_base;
    uint32_t inflight = 0;
    uint32_t trid = 1;
    uint32_t trid_to_wait = 1;
    for (uint32_t s = 0; s < nseg; ++s) {
        // (one bank, so one NOC xy: only the weight's bank-local base differs)
        const uint32_t base = (uint32_t)get_noc_addr_from_bank_id<true>(bank, get_arg_val<uint32_t>(3 + 3 * s));
        const uint32_t col0 = get_arg_val<uint32_t>(4 + 3 * s);
        const uint32_t ncols = get_arg_val<uint32_t>(5 + 3 * s);
        for (uint32_t c = 0; c < ncols; c += g) {
            const uint32_t gw = ncols - c < g ? ncols - c : g;
            for (uint32_t k0 = 0; k0 < kt; k0 += kb) {
                cb_reserve_back(cb_w, blk * (inflight + 1));
                noc_async_read_set_trid(trid);
                // column by column, k rows at the bank-page stride (consecutive pages of one bank read back to back
                // measured slower: 4-page reads 52 -> 62 us)
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
    noc_async_read_one_packet_set_state<true>(get_noc_addr(MEM_ZEROS_BASE), 0, 1);  // restore the read cmd buffer
    noc_async_read_set_trid(0);
}
