// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Packed decode all-reduce (tt/ccl.py): the T live rows of a [1, 1, 32, H] bf16 TILE tensor <-> K width-sharded tiles
// (the all-reduce's input / output). Chunk c (32 B: 16 values) = (tile c / 2T, row (c / 2) % T, face c % 2) of the
// tensor <-> byte 32 c of the packed tiles. Core j (row-major over the shard grid) owns packed tiles [j KJ, (j+1) KJ):
// PACK reads its chunks from the tensor into its shard, UNPACK writes its shard's chunks to the tensor.
//
// common args: tensor address, shard address (the same L1 address on every shard core)

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr uint32_t KJ = get_compile_time_arg_val(1);  // packed tiles per core
    constexpr uint32_t NCH = get_compile_time_arg_val(2);  // live chunks (2 T Ht)
    constexpr uint32_t CX = get_compile_time_arg_val(3);
    constexpr auto t_args = TensorAccessorArgs<4>();
    const uint32_t j = (uint32_t)get_absolute_logical_y() * CX + (uint32_t)get_absolute_logical_x();
    const auto at = TensorAccessor(t_args, get_common_arg_val<uint32_t>(0), 2048);
    const uint32_t shard = get_common_arg_val<uint32_t>(1);
    const uint32_t c0 = j * KJ * 64;
    const uint32_t c1 = c0 + KJ * 64 < NCH ? c0 + KJ * 64 : NCH;
#ifdef ARP_PACK
    // a DRAM source needs 64 B-aligned destinations matching the source: each touched tile's live prefix (faces 0 / 1,
    // rows < T) is read whole into the staging CB, then its chunks are copied L1 -> L1 into the shard
    constexpr uint32_t PREFIX = ((512 + 32 * T) + 63) / 64 * 64;
    cb_reserve_back(0, 1);
    const uint32_t stage = get_write_ptr(0);  // (this core's tiles' prefixes, PREFIX apart)
    const uint32_t tile0 = c0 / (2 * T);
    if (c0 < c1) {
        for (uint32_t tile = tile0; tile <= (c1 - 1) / (2 * T); ++tile) {
            noc_async_read(at.get_noc_addr(tile), stage + (tile - tile0) * PREFIX, PREFIX);
        }
        noc_async_read_barrier();
    }
    for (uint32_t c = c0; c < c1; ++c) {
        const uint32_t off = (c / (2 * T) - tile0) * PREFIX + (c & 1) * 512 + ((c >> 1) % T) * 32;
        noc_async_read(get_noc_addr(stage + off), shard + (c - c0) * 32, 32);
    }
#else
    for (uint32_t c = c0; c < c1; ++c) {
        const uint32_t off = (c & 1) * 512 + ((c >> 1) % T) * 32;
        noc_async_write(shard + (c - c0) * 32, at.get_noc_addr(c / (2 * T)) + off, 32);
    }
#endif
#ifdef ARP_PACK
    noc_async_read_barrier();
#else
    noc_async_write_barrier();
#endif
}
