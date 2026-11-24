// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Lane-aware score, compute (see ../__init__.py LaneSelOp.score). Per unit: P = relu(Qp @ K^T) for the packed query
// tile (rows 2 h + (r & 1)), one DEST pass of 4 dim tiles, packed bf16 with the packer ReLU; the writer scatters P's
// rows into 16 per-head tiles T_h (row r = the query's own row); then acc = sum_h T_h * w[:, h], one head per DEST
// pass, the heads accumulated in L1 by the packer (head 0 overwrites). This is indexer_score_dsa's decode configuration
// (q_chunk 32, k_chunk 32, head_group 1: compute_indexer_score.cpp accumulate_heads): the same LLK calls on the same
// operand values in the same order, so every score is bit-identical; only the rows packed per matmul differ.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"

constexpr uint32_t cb_k = 0, cb_q = 1, cb_w = 2, cb_p = 3, cb_t = 4, cb_acc = 5, cb_meta = 6;
constexpr uint32_t DT = 4;  // head_dim tiles

inline void set_matmul_mode() {
    reconfig_data_format(cb_t, cb_k, cb_w, cb_q);
    pack_reconfig_data_format(cb_acc, cb_p);
    pack_reconfig_l1_acc(0);
    matmul_block_init(cb_q, cb_k, 1 /*transpose k*/, 1 /*ct_dim*/, 1 /*rt_dim*/, DT /*kt_dim*/);
    pack_relu_config(ReluConfig::zero());
}

inline void set_mul_mode() {
    pack_relu_config(ReluConfig::none());
    reconfig_data_format(cb_k, cb_t, cb_q, cb_w);
    pack_reconfig_data_format(cb_p, cb_acc);
    mul_bcast_cols_init(cb_t, cb_w);
    MATH((llk_math_eltwise_binary_init<ckernel::EltwiseBinaryType::ELWMUL, ckernel::BroadcastType::COL, MATH_FIDELITY>(
        cb_t, cb_w, 1 /*acc_to_dest*/)));
}

void kernel_main() {
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q, cb_k, cb_p);
    cb_wait_front(cb_meta, 1);
    const uint32_t n = ckernel::read_tile_value(cb_meta, 0, 0);
    if (n == 0) {
        cb_pop_front(cb_meta, 1);
        return;
    }
    matmul_block_init(cb_q, cb_k, 1, 1, 1, DT);
    cb_wait_front(cb_w, 16);
    // software pipeline: P of unit u + 1 before the gate-mul of unit u (the writer scatters it meanwhile)
    bool have_q = false;
    auto mm = [&](uint32_t u) {
        if (ckernel::read_tile_value(cb_meta, 0, 1 + u)) {
            if (have_q) {
                cb_pop_front(cb_q, DT);
            }
            cb_wait_front(cb_q, DT);
            have_q = true;
        }
        cb_wait_front(cb_k, DT);
        set_matmul_mode();
        tile_regs_acquire();
        for (uint32_t d = 0; d < DT; ++d) {
            matmul_block(cb_q, cb_k, d, d, 0, 1 /*transpose k*/, 1, 1, DT);
        }
        tile_regs_commit();
        cb_reserve_back(cb_p, 1);
        tile_regs_wait();
        pack_block(0, cb_p, 1);
        tile_regs_release();
        cb_push_back(cb_p, 1);
        cb_pop_front(cb_k, DT);
    };
    mm(0);
    for (uint32_t u = 0; u < n; ++u) {
        if (u + 1 < n) {
            mm(u + 1);
        }
        cb_wait_front(cb_t, 16);
        cb_reserve_back(cb_acc, 1);
        set_mul_mode();
        for (uint32_t h = 0; h < 16; ++h) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(cb_t, cb_w, h, h, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_l1_acc(h == 0 ? 0 : 1);
            pack_tile<true>(0, cb_acc, 0);
            tile_regs_release();
        }
        pack_reconfig_l1_acc(0);
        cb_push_back(cb_acc, 1);
        cb_pop_front(cb_t, 16);
    }
    cb_pop_front(cb_q, DT);
    cb_pop_front(cb_w, 16);
    cb_pop_front(cb_meta, 1);
}
