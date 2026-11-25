// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Buffer-fed streamed decode projection, compute: kernels/ms_compute.cpp (same per-column K order, matmul_block, FP32
// dest) with a weight / output CB per segment. Segment kind 0: the prefetched weight from cb_wq (a local CB over the
// global CB: blocks of kbq k rows x gq tiles, the prefetcher's page), packed to cb_oq; kind 1: the DRAM-streamed weight
// from cb_w (blocks of kb x g), packed to cb_o.
//
// runtime args: [nseg, then per segment: kind, ncols]      compile args: cb_x, cb_wq, cb_w, cb_oq, cb_o, kt, kbq, gq, kb, g

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

template <uint32_t cb_wt, uint32_t kbt, uint32_t gt>
inline void segment(uint32_t cb_x, uint32_t cb_out, uint32_t kt, uint32_t x0, uint32_t ncols) {
    constexpr uint32_t blk = kbt * gt;
    uint32_t cur = 0;
    for (uint32_t c = 0; c < ncols; c += gt) {
        const uint32_t gw = ncols - c < gt ? ncols - c : gt;
        if (gw != cur) {
            matmul_block_init(cb_x, cb_wt, 0, gw, 1, kt);
            cur = gw;
        }
        tile_regs_acquire();
        for (uint32_t k0 = 0; k0 < kt; k0 += kbt) {
            cb_wait_front(cb_wt, blk);
            for (uint32_t k = 0; k < kbt; ++k) {
                matmul_block(cb_x, cb_wt, x0 + k0 + k, k * gt, 0, 0, gw, 1, kt);
            }
            cb_pop_front(cb_wt, blk);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < gw; ++j) {
            cb_reserve_back(cb_out, 1);
            pack_tile(j, cb_out);
            cb_push_back(cb_out, 1);
        }
        tile_regs_release();
    }
}

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_wq = get_compile_time_arg_val(1);
    constexpr uint32_t cb_w = get_compile_time_arg_val(2);
    constexpr uint32_t cb_oq = get_compile_time_arg_val(3);
    constexpr uint32_t cb_o = get_compile_time_arg_val(4);
    constexpr uint32_t kt = get_compile_time_arg_val(5);
    constexpr uint32_t kbq = get_compile_time_arg_val(6);
    constexpr uint32_t gq = get_compile_time_arg_val(7);
    constexpr uint32_t kb = get_compile_time_arg_val(8);
    constexpr uint32_t g = get_compile_time_arg_val(9);
    const uint32_t nseg = get_arg_val<uint32_t>(0);
    if (nseg == 0) {
        return;
    }
    const uint32_t kind0 = get_arg_val<uint32_t>(1);
    // matmul: in1 (weights) -> srcA, in0 -> srcB
    if (kind0 == 0) {
        compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_wq, cb_oq);
    } else {
        compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_o);
    }
    cb_wait_front(cb_x, kt * nseg);
    uint32_t prev = kind0;
    for (uint32_t s = 0; s < nseg; ++s) {
        const uint32_t kind = get_arg_val<uint32_t>(1 + 2 * s);
        const uint32_t ncols = get_arg_val<uint32_t>(2 + 2 * s);
        if (kind != prev) {
            pack_reconfig_data_format(kind == 0 ? cb_oq : cb_o);
            prev = kind;
        }
        if (kind == 0) {
            segment<cb_wq, kbq, gq>(cb_x, cb_oq, kt, s * kt, ncols);
        } else {
            segment<cb_w, kb, g>(cb_x, cb_o, kt, s * kt, ncols);
        }
    }
    cb_pop_front(cb_x, kt * nseg);
}
