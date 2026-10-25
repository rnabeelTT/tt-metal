// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode layer boundary, fabric sender (BRISC) (fabric_boundary/ops.py: FabricBoundary).
//
// The to-rows cores (fb_rows.cpp) pack this device's partial sum flat into its own slot of the site's receive buffer
// (value h at byte 2 h) and each increments program semaphore 1 here. Meanwhile the fabric connections open; once all
// `packers` have signalled, the slot is published into the same slot on every other device of the TP ring (from the
// gpt-oss fused decode sender, models/demos/gpt_oss/tt/fused_decode/kernels/boundary_sender.cpp): a fabric multicast
// of `fwd_hops` chips forward and `bwd_hops` chips backward, `chunk` bytes per packet, the last packet a fused write +
// receive-semaphore increment; then one local increment of the receive semaphore.
//
// runtime args: [unused, slot_addr, recv_noc_x, recv_noc_y, sem_addr, <fabric connection manager args>]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

void kernel_main() {
    const uint32_t partial_addr = get_arg_val<uint32_t>(0);
    const uint32_t slot_addr = get_arg_val<uint32_t>(1);
    const uint32_t recv_noc_x = get_arg_val<uint32_t>(2);
    const uint32_t recv_noc_y = get_arg_val<uint32_t>(3);
    const uint32_t sem_addr = get_arg_val<uint32_t>(4);

    constexpr uint32_t payload = get_compile_time_arg_val(0);
    constexpr uint32_t chunk = get_compile_time_arg_val(1);
    constexpr uint32_t fwd_hops = get_compile_time_arg_val(2);
    constexpr uint32_t bwd_hops = get_compile_time_arg_val(3);
    constexpr uint32_t packers = get_compile_time_arg_val(4);
    constexpr uint32_t chunks = (payload + chunk - 1) / chunk;

    size_t arg_idx = 5;
    auto fabric = FabricConnectionManager::build_from_args<
        FabricConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION_START_ONLY>(arg_idx);

    const uint32_t pack = slot_addr;
    PacketHeaderPool::reset();
    constexpr uint32_t ring_hdrs = chunks <= 4 ? chunks : 4;
    volatile PACKET_HEADER_TYPE* fwd_hdr[ring_hdrs];
    volatile PACKET_HEADER_TYPE* bwd_hdr[ring_hdrs];
    const uint64_t sem_noc = safe_get_noc_addr(recv_noc_x, recv_noc_y, sem_addr, 0);
    auto prepare = [&](uint32_t c, uint32_t h) {
        const uint32_t bytes = c + 1 < chunks ? chunk : payload - c * chunk;
        const uint64_t dst = safe_get_noc_addr(recv_noc_x, recv_noc_y, slot_addr + c * chunk, 0);
        if constexpr (fwd_hops > 0) {
            fwd_hdr[h]->to_chip_multicast(
                tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(fwd_hops)});
            if (c + 1 == chunks) {
                fwd_hdr[h]->to_noc_fused_unicast_write_atomic_inc(
                    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, sem_noc, 1, true}, bytes);
            } else {
                fwd_hdr[h]->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{dst}, bytes);
            }
        }
        if constexpr (bwd_hops > 0) {
            bwd_hdr[h]->to_chip_multicast(
                tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(bwd_hops)});
            if (c + 1 == chunks) {
                bwd_hdr[h]->to_noc_fused_unicast_write_atomic_inc(
                    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, sem_noc, 1, true}, bytes);
            } else {
                bwd_hdr[h]->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{dst}, bytes);
            }
        }
    };
    for (uint32_t h = 0; h < ring_hdrs; ++h) {
        if constexpr (fwd_hops > 0) {
            fwd_hdr[h] = PacketHeaderPool::allocate_header();
        }
        if constexpr (bwd_hops > 0) {
            bwd_hdr[h] = PacketHeaderPool::allocate_header();
        }
        prepare(h, h);
    }
    fabric.open_finish();
    {
        DeviceZoneScopedN("FB_PACKWAIT");
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1)), packers);
    }

    {
        DeviceZoneScopedN("FB_SEND");
        for (uint32_t c = 0; c < chunks; ++c) {
            const uint32_t bytes = c + 1 < chunks ? chunk : payload - c * chunk;
            const uint32_t src = pack + c * chunk;
            const uint32_t h = c % ring_hdrs;
            if (c >= ring_hdrs) {
                noc_async_writes_flushed();  // the header's previous packet has left L1
                prepare(c, h);
            }
            if constexpr (fwd_hops > 0) {
                auto& conn = fabric.get_forward_connection();
                conn.wait_for_empty_write_slot();
                conn.send_payload_without_header_non_blocking_from_address(src, bytes);
                conn.send_payload_flush_non_blocking_from_address((uint32_t)fwd_hdr[h], sizeof(PACKET_HEADER_TYPE));
            }
            if constexpr (bwd_hops > 0) {
                auto& conn = fabric.get_backward_connection();
                conn.wait_for_empty_write_slot();
                conn.send_payload_without_header_non_blocking_from_address(src, bytes);
                conn.send_payload_flush_non_blocking_from_address((uint32_t)bwd_hdr[h], sizeof(PACKET_HEADER_TYPE));
            }
        }
    }
    noc_semaphore_inc(get_noc_addr(sem_addr), 1);
    fabric.close();
    noc_async_atomic_barrier();
}
