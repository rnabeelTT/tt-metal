// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// Zero-copy host -> device expert writes for the hybrid MoE (hybrid/fetch.py): the device-format expert library lives
// in host memory pinned for the mesh and mapped to the NOC (tt-metal's experimental PinnedMemory), so a write of an
// expert into a device staging buffer is read by the device prefetcher straight from that memory (one host-DRAM read
// over PCIe), instead of the default path's CPU memcpy into the command-queue hugepage (two more DRAM passes and a
// busy core). A C API for ctypes: the MeshDevice comes from the ttnn Python object (hybrid/pin/nbptr.py).
//
// Build (host gcc, against the serving image's tt-metalium headers): hybrid/pin/build.sh
#include <cstdint>
#include <memory>
#include <vector>

#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/shard_data_transfer.hpp>

namespace ttm = tt::tt_metal;
namespace dist = tt::tt_metal::distributed;

namespace {
struct Region {
    std::shared_ptr<ttm::experimental::PinnedMemory> pinned;
    std::byte* base;
    uint64_t size;
};
std::vector<Region> g_regions;
std::vector<std::shared_ptr<dist::MeshBuffer>> g_dsts;

dist::MeshDevice& mesh_of(void* p) { return *static_cast<dist::MeshDevice*>(p); }

std::vector<dist::MeshCoordinate> coords(dist::MeshDevice& m) {
    std::vector<dist::MeshCoordinate> out;
    for (const auto& c : dist::MeshCoordinateRange(m.shape())) {
        out.push_back(c);
    }
    return out;
}
}  // namespace

extern "C" {

// Pins [host, host + size) for every device of the mesh (NOC-mapped). Returns the region id, or -1 when the system
// cannot map it to the NOC (the writes would then fall back to the hugepage path).
int pinx_pin(void* mesh, void* host, uint64_t size) {
    auto& m = mesh_of(mesh);
    auto* base = static_cast<std::byte*>(host);
    ttm::HostBuffer hb(ttsl::Span<std::byte>(base, size), ttm::MemoryPin());
    auto pm = ttm::experimental::PinnedMemory::Create(
        m, dist::MeshCoordinateRangeSet(dist::MeshCoordinateRange(m.shape())), hb, /*map_to_noc=*/true);
    for (const auto& c : coords(m)) {
        if (!pm->usable_from_noc(m.get_device(c)->id())) {
            return -1;
        }
    }
    g_regions.push_back({pm, base, size});
    return static_cast<int>(g_regions.size() - 1);
}

// A MeshBuffer view of an existing interleaved DRAM tensor (the same address on every device, size bytes each).
int pinx_dst(void* mesh, uint64_t address, uint64_t size, uint64_t page_size) {
    auto& m = mesh_of(mesh);
    dist::ReplicatedBufferConfig rc{.size = size};
    dist::DeviceLocalBufferConfig dl{.page_size = page_size, .buffer_type = ttm::BufferType::DRAM};
    g_dsts.push_back(dist::MeshBuffer::create(rc, dl, &m, address));
    return static_cast<int>(g_dsts.size() - 1);
}

// Device d of the mesh (row-major) <- region bytes [offsets[d], offsets[d] + size) into the destination's device-local
// bytes [dst_offset, dst_offset + size) (size 0: the whole destination; dst_offset a multiple of its page size).
// Non-blocking: the region must not change until the write completed.
int pinx_write_at(
    void* mesh, int dst, int region, const uint64_t* offsets, int cq_id, uint64_t dst_offset, uint64_t size) {
    auto& m = mesh_of(mesh);
    const auto& r = g_regions.at(region);
    const auto& buf = g_dsts.at(dst);
    const uint64_t n = size ? size : buf->device_local_size();
    if (dst_offset + n > buf->device_local_size() || dst_offset % buf->page_size() != 0) {
        return -3;
    }
    std::vector<dist::ShardDataTransfer> xfers;
    int i = 0;
    for (const auto& c : coords(m)) {
        if (offsets[i] + n > r.size) {
            return -2;
        }
        auto x = dist::ShardDataTransfer{c}.host_data(r.base + offsets[i]).region(ttm::BufferRegion(dst_offset, n));
        ttm::experimental::ShardDataTransferSetPinnedMemory(x, r.pinned);
        xfers.push_back(std::move(x));
        ++i;
    }
    m.mesh_command_queue(cq_id).enqueue_write_shards(buf, xfers, /*blocking=*/false);
    return 0;
}

int pinx_write(void* mesh, int dst, int region, const uint64_t* offsets, int cq_id) {
    return pinx_write_at(mesh, dst, region, offsets, cq_id, 0, 0);
}

// Region bytes [offsets[d], ..) <- device d's destination buffer (raw device bytes, e.g. bfp4 tiles). Blocking.
int pinx_read(void* mesh, int dst, int region, const uint64_t* offsets) {
    auto& m = mesh_of(mesh);
    const auto& r = g_regions.at(region);
    const auto& buf = g_dsts.at(dst);
    const uint64_t n = buf->device_local_size();
    std::vector<dist::ShardDataTransfer> xfers;
    int i = 0;
    for (const auto& c : coords(m)) {
        if (offsets[i] + n > r.size) {
            return -2;
        }
        xfers.push_back(dist::ShardDataTransfer{c}.host_data(r.base + offsets[i]).region(ttm::BufferRegion(0, n)));
        ++i;
    }
    m.mesh_command_queue(0).enqueue_read_shards(xfers, buf, /*blocking=*/true);
    return 0;
}

// Region bytes [offsets[d], + size) <- device d's destination bytes [src_offset, + size) (a multiple of its page size).
// Blocking, on command queue cq_id.
int pinx_read_at(
    void* mesh, int dst, int region, const uint64_t* offsets, int cq_id, uint64_t src_offset, uint64_t size) {
    auto& m = mesh_of(mesh);
    const auto& r = g_regions.at(region);
    const auto& buf = g_dsts.at(dst);
    if (src_offset + size > buf->device_local_size() || src_offset % buf->page_size() != 0) {
        return -3;
    }
    std::vector<dist::ShardDataTransfer> xfers;
    int i = 0;
    for (const auto& c : coords(m)) {
        if (offsets[i] + size > r.size) {
            return -2;
        }
        xfers.push_back(
            dist::ShardDataTransfer{c}.host_data(r.base + offsets[i]).region(ttm::BufferRegion(src_offset, size)));
        ++i;
    }
    m.mesh_command_queue(cq_id).enqueue_read_shards(xfers, buf, /*blocking=*/true);
    return 0;
}

uint64_t pinx_dst_size(int dst) { return g_dsts.at(dst)->device_local_size(); }
}

extern "C" {
// Drops every destination view and pinned region; call before closing the mesh (their destructors talk to it).
void pinx_release() {
    g_dsts.clear();
    g_regions.clear();
}
}

// GCC >= 13 headers make <iostream> users call std::ios_base_library_init (GLIBCXX_3.4.32), which the serving image's
// older libstdc++ lacks; its libstdc++ initializes the standard streams itself, so a local no-op satisfies the call.
namespace std {
void ios_base_library_init() {}
}  // namespace std
