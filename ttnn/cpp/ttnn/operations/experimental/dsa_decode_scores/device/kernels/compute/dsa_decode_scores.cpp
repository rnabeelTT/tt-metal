// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Per active unit (slot s, key block j): sc = relu(q_s k_j^T) [32 heads x 32 keys] (bf16 operands, fp32 DST), then
// the head-weighted row w_s . sc [1 x 32] (w_s in row 0 of an fp32 tile, rows 1..31 zero).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/cb_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/dataflow/circular_buffer.h"

namespace {
constexpr uint32_t cb_q = 0, cb_w = 1, cb_k = 2, cb_ctrl = 3, cb_sc = 6, cb_out = 16;
}  // namespace

void kernel_main() {
    constexpr uint32_t Dt = get_compile_time_arg_val(0);
    constexpr uint32_t nblk = get_compile_time_arg_val(1);
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    if (count == 0) {
        return;
    }
    compute_kernel_hw_startup(cb_q, cb_k, cb_sc);
    CircularBuffer q(cb_q), w(cb_w), k(cb_k), ctrl(cb_ctrl), sc(cb_sc), out(cb_out);

    uint32_t cur = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t u = start + i;
        const uint32_t s = u / nblk;
        if (s != cur) {
            if (cur != 0xFFFFFFFFu) {
                q.pop_front(Dt);
                w.pop_front(1);
            }
            cur = s;
            q.wait_front(Dt);
            w.wait_front(1);
        }
        ctrl.wait_front(1);
        const uint32_t active = ckernel::read_tile_value(cb_ctrl, /*tile=*/0, /*element_offset=*/0);
        ctrl.pop_front(1);
        if (!active) {
            continue;
        }
        k.wait_front(Dt);
        sc.reserve_back(1);
        // explicit unpack reconfig: bf16 q / k here, fp32 w / sc below (matmul_init does not reconfigure formats)
        reconfig_data_format(cb_q, cb_k);
        tile_regs_acquire();
        matmul_init(cb_q, cb_k, /*transpose=*/1);
        for (uint32_t d = 0; d < Dt; ++d) {
            matmul_tiles(cb_q, cb_k, d, d, 0);
        }
        relu_tile_init();
        relu_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_sc);
        tile_regs_release();
        sc.push_back(1);
        k.pop_front(Dt);

        sc.wait_front(1);
        out.reserve_back(1);
        reconfig_data_format(cb_w, cb_sc);
        tile_regs_acquire();
        matmul_init(cb_w, cb_sc, /*transpose=*/0);
        matmul_tiles(cb_w, cb_sc, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        out.push_back(1);
        sc.pop_front(1);
    }
    q.pop_front(Dt);
    w.pop_front(1);
}
