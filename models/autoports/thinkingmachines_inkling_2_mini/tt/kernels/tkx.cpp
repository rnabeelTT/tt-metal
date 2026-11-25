// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The sampling tail's die exchange on a vocab-sharded head (tt/sampling.py TailX, I2M_TOPK_X), one core per die: this
// die's payload goes as one fused fabric write + increment (flushed: the data lands first) into slot ME of a persistent
// DRAM scratch on every other die (a multicast FWD dies forward and BWD backward), then this die waits for the N - 1
// other slots and builds its output from the N payloads in die order (ttnn.all_gather's order).
// TKX_MAX: the payload is the L rows' fp32 shifts (column 0 of rows 0 .. L - 1 of one tile; DENSE: of row 0 of tile r,
//   which holds it in all 32 rows); out = the max over the dies of each row, in the input's layout, as the all-gather +
//   ttnn.max it replaces.
// else: the payload is the top-k's index tile (IB bytes per entry) + value tile; out = [rows, N K] index / value tensors
//   (die d's tiles at tile d), as the two all-gathers it replaces.
//
// compile args: cb, N, ME, L, DENSE, this core's noc x / y, accessors (in0, in1, scratch, out0, out1)
// runtime args: in0, in1, scratch, out0, out1 addresses, global semaphore address, <fabric connection manager args>

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

static inline uint32_t key_of(uint32_t u) { return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t N = get_compile_time_arg_val(1);
    constexpr uint32_t ME = get_compile_time_arg_val(2);
    constexpr uint32_t FWD = N - 1 - ME, BWD = ME;
    constexpr uint32_t L = get_compile_time_arg_val(3);     // TKX_MAX: live rows; else IB (index bytes)
    constexpr uint32_t DENSE = get_compile_time_arg_val(4);  // TKX_MAX: [1, L, 32, 1] shift tiles (RowPack)
    constexpr uint32_t SX = get_compile_time_arg_val(5), SY = get_compile_time_arg_val(6);  // this core (noc)
    constexpr auto a0 = TensorAccessorArgs<7>();
    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
    constexpr auto as = TensorAccessorArgs<a1.next_compile_time_args_offset()>();
    constexpr auto o0 = TensorAccessorArgs<as.next_compile_time_args_offset()>();
    constexpr auto o1 = TensorAccessorArgs<o0.next_compile_time_args_offset()>();
#if TKX_MAX
    constexpr uint32_t P0 = 4096, P1 = 0;  // input / output tile bytes
    constexpr uint32_t SLOT = 64;          // payload: L fp32
    constexpr uint32_t CH = 1;             // fabric packets per payload
#else
    constexpr uint32_t P0 = 1024 * L, P1 = 2048;
    constexpr uint32_t SLOT = P0 + P1;
    constexpr uint32_t CH = SLOT <= 4096 ? 1 : 2;  // (packets <= 4352 B: the index tile, then the value tile)
#endif
    const uint32_t in0 = get_arg_val<uint32_t>(0), in1 = get_arg_val<uint32_t>(1), scr = get_arg_val<uint32_t>(2);
    const uint32_t out0 = get_arg_val<uint32_t>(3), out1 = get_arg_val<uint32_t>(4);
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(5));
    size_t arg_idx = 6;
    auto fabric = FabricConnectionManager::build_from_args<
        FabricConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION_START_ONLY>(arg_idx);

    const auto s0 = TensorAccessor(a0, in0, P0);
    const auto s_scr = TensorAccessor(as, scr, SLOT);
    // L1: N slots (die d's payload at d * SLOT), then (TKX_MAX) the output tile
    const uint32_t buf = get_write_ptr(cb);
    const uint32_t mine = buf + ME * SLOT;

    PacketHeaderPool::reset();
    volatile PACKET_HEADER_TYPE* fh[CH];
    volatile PACKET_HEADER_TYPE* bh[CH];
    for (uint32_t c = 0; c < CH; ++c) {
        const uint32_t off = c == 0 ? 0 : P0;
        const uint32_t bytes = CH == 1 ? SLOT : (c == 0 ? P0 : P1);
        const uint64_t dst = s_scr.get_noc_addr(ME, off);
        const uint64_t sa = safe_get_noc_addr(SX, SY, reinterpret_cast<uint32_t>(sem), 0);
        if constexpr (FWD > 0) {
            fh[c] = PacketHeaderPool::allocate_header();
            fh[c]->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(FWD)});
            fh[c]->to_noc_fused_unicast_write_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, sa, 1, true}, bytes);
        }
        if constexpr (BWD > 0) {
            bh[c] = PacketHeaderPool::allocate_header();
            bh[c]->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(BWD)});
            bh[c]->to_noc_fused_unicast_write_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, sa, 1, true}, bytes);
        }
    }

    // this die's payload
#if TKX_MAX
    const uint32_t tile = buf + N * SLOT;
    if constexpr (DENSE) {  // row r's value: element 0 of tile r
        for (uint32_t r = 0; r < L; ++r) {
            noc_async_read(s0.get_noc_addr(r), tile + r * 16, 16);
        }
    } else {
        noc_async_read(s0.get_noc_addr(0), tile, P0);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    {
        volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile);
        volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mine);
        for (uint32_t r = 0; r < L; ++r) {
            m[r] = DENSE ? t[r * 4] : t[(r >> 4) * 512 + (r & 15) * 16];
        }
    }
#else
    const auto s1 = TensorAccessor(a1, in1, P1);
    noc_async_read(s0.get_noc_addr(0), mine, P0);
    noc_async_read(s1.get_noc_addr(0), mine + P0, P1);
    noc_async_read_barrier();
#endif
    fabric.open_finish();
    for (uint32_t c = 0; c < CH; ++c) {
        const uint32_t src = mine + (c == 0 ? 0 : P0);
        const uint32_t bytes = CH == 1 ? SLOT : (c == 0 ? P0 : P1);
        if constexpr (FWD > 0) {
            auto& conn = fabric.get_forward_connection();
            conn.wait_for_empty_write_slot();
            conn.send_payload_without_header_non_blocking_from_address(src, bytes);
            conn.send_payload_flush_non_blocking_from_address((uint32_t)fh[c], sizeof(PACKET_HEADER_TYPE));
        }
        if constexpr (BWD > 0) {
            auto& conn = fabric.get_backward_connection();
            conn.wait_for_empty_write_slot();
            conn.send_payload_without_header_non_blocking_from_address(src, bytes);
            conn.send_payload_flush_non_blocking_from_address((uint32_t)bh[c], sizeof(PACKET_HEADER_TYPE));
        }
    }
#if !TKX_MAX
    {  // this die's own tiles straight to its outputs while the others arrive
        const auto d0 = TensorAccessor(o0, out0, P0);
        const auto d1 = TensorAccessor(o1, out1, P1);
        noc_async_write(mine, d0.get_noc_addr(ME), P0);
        noc_async_write(mine + P0, d1.get_noc_addr(ME), P1);
    }
#endif
    fabric.close();

    noc_semaphore_wait(sem, (N - 1) * CH);
    noc_semaphore_set(sem, 0);
    for (uint32_t d = 0; d < N; ++d) {
        if (d != ME) {
            noc_async_read(s_scr.get_noc_addr(d), buf + d * SLOT, SLOT);
        }
    }
    noc_async_read_barrier();
#if TKX_MAX
    {
        invalidate_l1_cache();
        const auto d0 = TensorAccessor(o0, out0, P0);
        uint32_t best[L];
        for (uint32_t r = 0; r < L; ++r) {
            uint32_t bk = 0;
            for (uint32_t d = 0; d < N; ++d) {
                const uint32_t u = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf + d * SLOT)[r];
                const uint32_t k = key_of(u);
                if (d == 0 || k > bk) {
                    bk = k;
                    best[r] = u;
                }
            }
        }
        // DENSE: tile r holds row r's value in column 0 of all 32 rows (L tiles); else rows 0 .. L - 1 of one tile
        for (uint32_t o = 0; o < (DENSE ? L : 1); ++o) {
            volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile + o * P0);
            for (uint32_t i = 0; i < 1024; ++i) {
                t[i] = 0;
            }
            for (uint32_t r = 0; r < (DENSE ? 32 : L); ++r) {
                t[(r >> 4) * 512 + (r & 15) * 16] = DENSE ? best[o] : best[r];
            }
            noc_async_write(tile + o * P0, d0.get_noc_addr(o), P0);
        }
    }
#else
    {
        const auto d0 = TensorAccessor(o0, out0, P0);
        const auto d1 = TensorAccessor(o1, out1, P1);
        for (uint32_t d = 0; d < N; ++d) {
            if (d != ME) {
                noc_async_write(buf + d * SLOT, d0.get_noc_addr(d), P0);
                noc_async_write(buf + d * SLOT + P0, d1.get_noc_addr(d), P1);
            }
        }
    }
#endif
    noc_async_write_barrier();
}
