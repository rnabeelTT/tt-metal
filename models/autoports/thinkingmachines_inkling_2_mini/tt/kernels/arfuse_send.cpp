// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fused packed decode all-reduce (tt/ccl.py _Packed.fused), fabric sender of one link: once the packers
// (arfuse_pack.cpp) have put this die's slots of its NC shard cores into the payload CB, each slot goes to the same slot of that shard core
// on every other die of the line (a multicast FWD dies forward and BWD backward), as a fused write + increment of that
// core's global semaphore (fabric sender: qwen36 fabric_boundary/kernels/fb_sender.cpp).
//
// runtime args: slot address, global semaphore address, NC, packer count, NC noc x, NC noc y, <fabric connection manager args>

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

void kernel_main() {
    constexpr uint32_t BYTES = get_compile_time_arg_val(0);  // one shard core's slot
    constexpr uint32_t FWD = get_compile_time_arg_val(1);
    constexpr uint32_t BWD = get_compile_time_arg_val(2);
    constexpr uint32_t cb_tx = get_compile_time_arg_val(3);

    const uint32_t slot = get_arg_val<uint32_t>(0);
    const uint32_t sem_addr = get_arg_val<uint32_t>(1);
    const uint32_t NC = get_arg_val<uint32_t>(2);
    const uint32_t packers = get_arg_val<uint32_t>(3);
    size_t arg_idx = 4;
    const uint32_t xs = arg_idx;
    arg_idx += NC;
    const uint32_t ys = arg_idx;
    arg_idx += NC;
    auto fabric = FabricConnectionManager::build_from_args<
        FabricConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION_START_ONLY>(arg_idx);

    PacketHeaderPool::reset();
    constexpr uint32_t HMAX = 4;
    const uint32_t H = NC <= HMAX ? NC : HMAX;
    volatile PACKET_HEADER_TYPE* fwd_hdr[HMAX];
    volatile PACKET_HEADER_TYPE* bwd_hdr[HMAX];
    auto prepare = [&](uint32_t c, uint32_t h) {
        const uint32_t x = get_arg_val<uint32_t>(xs + c), y = get_arg_val<uint32_t>(ys + c);
        const uint64_t dst = safe_get_noc_addr(x, y, slot, 0);
        const uint64_t sem = safe_get_noc_addr(x, y, sem_addr, 0);
        if constexpr (FWD > 0) {
            fwd_hdr[h]->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(FWD)});
            fwd_hdr[h]->to_noc_fused_unicast_write_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, sem, 1, false}, BYTES);
        }
        if constexpr (BWD > 0) {
            bwd_hdr[h]->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(BWD)});
            bwd_hdr[h]->to_noc_fused_unicast_write_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, sem, 1, false}, BYTES);
        }
    };
    for (uint32_t h = 0; h < H; ++h) {
        if constexpr (FWD > 0) {
            fwd_hdr[h] = PacketHeaderPool::allocate_header();
        }
        if constexpr (BWD > 0) {
            bwd_hdr[h] = PacketHeaderPool::allocate_header();
        }
        prepare(h, h);
    }
    fabric.open_finish();
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0)), packers);

    const uint32_t tx = get_write_ptr(cb_tx);
    for (uint32_t c = 0; c < NC; ++c) {
        const uint32_t h = c % H;
        if (c >= H) {
            noc_async_writes_flushed();  // (the header's previous packet has left L1)
            prepare(c, h);
        }
        const uint32_t src = tx + c * BYTES;
        if constexpr (FWD > 0) {
            auto& conn = fabric.get_forward_connection();
            conn.wait_for_empty_write_slot();
            conn.send_payload_without_header_non_blocking_from_address(src, BYTES);
            conn.send_payload_flush_non_blocking_from_address((uint32_t)fwd_hdr[h], sizeof(PACKET_HEADER_TYPE));
        }
        if constexpr (BWD > 0) {
            auto& conn = fabric.get_backward_connection();
            conn.wait_for_empty_write_slot();
            conn.send_payload_without_header_non_blocking_from_address(src, BYTES);
            conn.send_payload_flush_non_blocking_from_address((uint32_t)bwd_hdr[h], sizeof(PACKET_HEADER_TYPE));
        }
    }
    fabric.close();
    noc_async_write_barrier();
}
