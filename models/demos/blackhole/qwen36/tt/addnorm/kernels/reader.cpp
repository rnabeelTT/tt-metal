// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Decode add + RMSNorm, reader (one core per residual shard): this core's T tiles of the residual (and of the all-reduce
// result when ADD) are its CBs (pushed in place), its T (gamma) tiles (every row = gamma) and the all-ones tile.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t ones_addr = get_arg_val<uint32_t>(1);
    const uint32_t shard = get_arg_val<uint32_t>(2);

    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr uint32_t ADD = get_compile_time_arg_val(1);
    constexpr uint32_t cb_res = get_compile_time_arg_val(2);
    constexpr uint32_t cb_red = get_compile_time_arg_val(3);
    constexpr uint32_t cb_w = get_compile_time_arg_val(4);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(5);
    constexpr auto w_args = TensorAccessorArgs<6>();
    constexpr auto ones_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr uint32_t BF_TILE = 2048;
    const auto s_w = TensorAccessor(w_args, w_addr, BF_TILE);
    const auto s_ones = TensorAccessor(ones_args, ones_addr, BF_TILE);

    DeviceZoneScopedN("rd");
    // the local shards (CBs on the tensors themselves) first, then the DRAM gamma / ones tiles
    cb_reserve_back(cb_res, T);
    cb_push_back(cb_res, T);
    if constexpr (ADD) {
        cb_reserve_back(cb_red, T);
        cb_push_back(cb_red, T);
    }
    cb_reserve_back(cb_ones, 1);
    noc_async_read(s_ones.get_noc_addr(0), get_write_ptr(cb_ones), BF_TILE);
    cb_reserve_back(cb_w, T);
    const uint32_t w = get_write_ptr(cb_w);
    for (uint32_t t = 0; t < T; ++t) {
        noc_async_read(s_w.get_noc_addr(shard * T + t), w + t * BF_TILE, BF_TILE);
    }
    noc_async_read_barrier();
    cb_push_back(cb_ones, 1);
    cb_push_back(cb_w, T);
}
