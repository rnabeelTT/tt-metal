// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Prefill expert SwiGLU, compute (bit-identical to the ttnn chain it replaces, 16-bit DEST like those ops):
//   g  = clamp(gate, -inf, L)           ttnn.clamp(max=L)        -> clamp_tile (CLAMP_TSS, bf16 bits as ttnn passes)
//   u  = clamp(up, -L, L)               ttnn.clamp               -> clamp_tile
//   d  = u * silu(g)                    ttnn.multiply(u, g, input_tensor_b_activations=[SILU]): binary_ng SFPU MUL
//                                       (bf16, accurate mode) with the rhs SILU preprocess -> silu_tile, mul_binary_tile
//   o  = d * bcast_col(w)               ttnn.multiply(d [Rd, Il], w [Rd, 1]): binary_ng SFPU MUL, COL broadcast
//                                       (unary_bcast<COL> of the weight tile) -> unary_bcast, mul_binary_tile
// Every intermediate of the ttnn chain is a bf16 tile packed from / unpacked to a 16-bit DEST (exact both ways), so
// keeping them in DEST gives the same bits. Compile args: NT (tiles per half), L bits, -L bits, -inf bits.
#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t NT = get_compile_time_arg_val(0);
    constexpr uint32_t LIM = get_compile_time_arg_val(1);
    constexpr uint32_t NLIM = get_compile_time_arg_val(2);
    constexpr uint32_t NINF = get_compile_time_arg_val(3);
    constexpr uint32_t B = 3;  // tile columns per DEST pass: u in 0..B-1, g in B..2B-1, the weight column in 2B
    static_assert(NT % B == 0);
    constexpr uint32_t cb_in = 0, cb_w = 1, cb_out = 16;
    const uint32_t ng = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_in, cb_out);
    for (uint32_t g = 0; g < ng; ++g) {
        cb_wait_front(cb_in, 2 * NT);
        cb_wait_front(cb_w, 1);
        for (uint32_t j0 = 0; j0 < NT; j0 += B) {
            cb_reserve_back(cb_out, B);
            tile_regs_acquire();
            copy_init(cb_in);
            for (uint32_t i = 0; i < B; ++i) {
                copy_tile(cb_in, j0 + i, i);           // up
                copy_tile(cb_in, NT + j0 + i, B + i);  // gate
            }
            clamp_tile_init();
            for (uint32_t i = 0; i < B; ++i) {
                clamp_tile(B + i, NINF, LIM);
            }
            for (uint32_t i = 0; i < B; ++i) {
                clamp_tile(i, NLIM, LIM);
            }
            silu_tile_init();
            for (uint32_t i = 0; i < B; ++i) {
                silu_tile(B + i);
            }
            mul_binary_tile_init();
            for (uint32_t i = 0; i < B; ++i) {
                mul_binary_tile(i, B + i, i);
            }
            unary_bcast_init<BroadcastType::COL>(cb_w);
            unary_bcast<BroadcastType::COL>(cb_w, 0, 2 * B);
            unary_bcast_uninit<BroadcastType::COL>(cb_w);
            mul_binary_tile_init();
            for (uint32_t i = 0; i < B; ++i) {
                mul_binary_tile(i, 2 * B, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < B; ++i) {
                pack_tile(i, cb_out);
            }
            tile_regs_release();
            cb_push_back(cb_out, B);
        }
        cb_pop_front(cb_in, 2 * NT);
        cb_pop_front(cb_w, 1);
    }
}
