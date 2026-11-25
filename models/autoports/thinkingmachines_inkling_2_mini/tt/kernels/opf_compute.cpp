// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Buffer-fed decode o_proj, compute: the prefetched weight (cb_wq: pages of kbq k rows x npr tiles, k-major) times the
// activation (cb_x, kt tiles), in ttnn's 1D decode matmul order (bmm_large_block_zm_fused_bias_activation with
// FP32_DEST_ACC_EN + PACKER_L1_ACC): K blocks of lin tiles accumulated in dest; blocks 0 .. nb - 2 packed into an FP32
// partial with packer L1 accumulation, the last block on top of the partial reloaded into dest (unpack to dest FP32),
// then packed to cb_out (bitwise ttnn.linear with in0_block_w = lin).
//
// runtime args: [on]      compile args: cb_x, cb_wq, cb_part, cb_out, kt, kbq, npr, lin

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_wq = get_compile_time_arg_val(1);
    constexpr uint32_t cb_part = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t kt = get_compile_time_arg_val(4);
    constexpr uint32_t kbq = get_compile_time_arg_val(5);
    constexpr uint32_t npr = get_compile_time_arg_val(6);
    constexpr uint32_t lin = get_compile_time_arg_val(7);
    constexpr uint32_t nb = kt / lin;
    constexpr uint32_t page = kbq * npr;
    static_assert(kt % lin == 0 && kt % kbq == 0 && npr <= 4);
    if (get_arg_val<uint32_t>(0) == 0) {
        return;
    }
    // matmul: in1 (weights) -> srcA, in0 -> srcB
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_wq, nb > 1 ? cb_part : cb_out);
    matmul_block_init(cb_x, cb_wq, 0, npr, 1, lin);
    cb_wait_front(cb_x, kt);
    for (uint32_t b = 0; b < nb; ++b) {
        const bool last = b == nb - 1;
        tile_regs_acquire();
        if (last && nb > 1) {  // (reload_from_cb_to_dst)
            reconfig_data_format_srca(cb_wq, cb_part);
            copy_init(cb_part);
            cb_wait_front(cb_part, npr);
            for (uint32_t j = 0; j < npr; ++j) {
                copy_tile(cb_part, j, j);
            }
            cb_pop_front(cb_part, npr);
            reconfig_data_format_srca(cb_part, cb_wq);
            matmul_block_init(cb_x, cb_wq, 0, npr, 1, lin);
        }
        for (uint32_t i = 0; i < lin; ++i) {
            const uint32_t k = b * lin + i;
            const uint32_t kp = k % kbq;
            if (kp == 0) {
                cb_wait_front(cb_wq, page);
            }
            matmul_block(cb_x, cb_wq, k, kp * npr, 0, 0, npr, 1, lin);
            if (kp == kbq - 1) {
                cb_pop_front(cb_wq, page);
            }
        }
        tile_regs_commit();
        if (last) {
            cb_reserve_back(cb_out, npr);
            tile_regs_wait();
            if constexpr (nb > 1) {
                PACK((pack_reconfig_data_format(cb_out)));
                PACK((llk_pack_reconfig_l1_acc(0)));
            }
            for (uint32_t j = 0; j < npr; ++j) {
                pack_tile(j, cb_out);
            }
            tile_regs_release();
            cb_push_back(cb_out, npr);
        } else {
            cb_reserve_back(cb_part, npr);
            tile_regs_wait();
            if (b == 0) {
                PACK((llk_pack_reconfig_l1_acc(0)));
            } else if (b == 1) {
                PACK((llk_pack_reconfig_l1_acc(1)));
            }
            for (uint32_t j = 0; j < npr; ++j) {
                pack_tile(j, cb_part);
            }
            tile_regs_release();
            cb_push_back(cb_part, npr);
            if (b + 2 < nb) {
                cb_wait_front(cb_part, npr);
                cb_pop_front(cb_part, npr);
            }
        }
    }
    cb_pop_front(cb_x, kt);
}
