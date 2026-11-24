// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Streamed decode projection, compute (tt/mstream.py; the LM head's kernels/head_compute.cpp without the softcap, with
// column groups and segments): per group of up to g output columns, sum over the kt K tiles of x[k] @ W[k, column]
// into g FP32 DST tiles (matmul_block, one activation tile per k for the whole group), packed in column order. The
// weight arrives in blocks of kb k rows x g tiles (kernels/ilstream_reader.cpp). Segment s's columns use the activation
// at cb_x tile s kt.
//
// runtime args: [nseg, then per segment: ncols]      compile args: cb_x, cb_w, cb_out, kt, kb, g

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t kt = get_compile_time_arg_val(3);
    constexpr uint32_t kb = get_compile_time_arg_val(4);
    constexpr uint32_t g = get_compile_time_arg_val(5);
    constexpr uint32_t blk = kb * g;
    const uint32_t nseg = get_arg_val<uint32_t>(0);
    if (nseg == 0) {
        return;
    }
    // matmul: in1 (weights) -> srcA, in0 -> srcB (formats and tile shapes: the activation CB may hold h x 32 tiles)
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);
    cb_wait_front(cb_x, kt * nseg);
    uint32_t cur = 0;
    for (uint32_t s = 0; s < nseg; ++s) {
        const uint32_t ncols = get_arg_val<uint32_t>(1 + s);
        const uint32_t x0 = s * kt;
        for (uint32_t c = 0; c < ncols; c += g) {
            const uint32_t gw = ncols - c < g ? ncols - c : g;
            if (gw != cur) {
                matmul_block_init(cb_x, cb_w, 0, gw, 1, kt);
                cur = gw;
            }
            tile_regs_acquire();
            for (uint32_t k0 = 0; k0 < kt; k0 += kb) {
                cb_wait_front(cb_w, blk);
                for (uint32_t k = 0; k < kb; ++k) {
                    matmul_block(cb_x, cb_w, x0 + k0 + k, k * g, 0, 0, gw, 1, kt);
                }
                cb_pop_front(cb_w, blk);
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
    cb_pop_front(cb_x, kt * nseg);
}
