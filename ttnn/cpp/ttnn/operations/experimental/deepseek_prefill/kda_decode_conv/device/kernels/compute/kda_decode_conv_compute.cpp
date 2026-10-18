// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// KDA decode conv step per tile column (all slots in one tile row), on the SFPU in fp32 DST:
//   acc = cur * t3 + h0 * t0 + h1 * t1 + h2 * t2;  qkv = where(active, silu(acc), 0)
//   h0' = where(active, h1, h0);  h1' = where(active, h2, h1);  h2' = where(active, cur, h2)
// bf16 tiles (cur, history) unpack into fp32 DST exactly; the fp32 taps / mask unpack straight to DST
// (UnpackToDestFp32, no TF32 truncation). The mask and tap broadcasts are done by the reader.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

namespace {
constexpr uint32_t CB_CUR = get_compile_time_arg_val(0);
constexpr uint32_t CB_H = get_compile_time_arg_val(1);
constexpr uint32_t CB_TAP = get_compile_time_arg_val(2);
constexpr uint32_t CB_ACT = get_compile_time_arg_val(3);
constexpr uint32_t CB_QKV = get_compile_time_arg_val(4);
constexpr uint32_t CB_HOUT = get_compile_time_arg_val(5);

ALWI void load(uint32_t cb, uint32_t idx, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, idx, dst);
}

ALWI void select(uint32_t cond, uint32_t t, uint32_t f, uint32_t out) {
    where_tile_init();
    where_tile<DataFormat::Float32>(cond, t, f, out);
}
}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(CB_CUR, CB_TAP, CB_QKV);
    const uint32_t n = get_arg_val<uint32_t>(0);
    CircularBuffer cc(CB_CUR), ch(CB_H), ct(CB_TAP), ca(CB_ACT), cq(CB_QKV), co(CB_HOUT);
    ca.wait_front(1);  // resident slot mask
    for (uint32_t i = 0; i < n; ++i) {
        cc.wait_front(1);
        ch.wait_front(3);
        ct.wait_front(4);

        // ---- qkv
        tile_regs_acquire();
        load(CB_CUR, 0, 0);
        load(CB_TAP, 3, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        for (uint32_t j = 0; j < 3; ++j) {
            load(CB_H, j, 1);
            load(CB_TAP, j, 2);
            mul_binary_tile_init();
            mul_binary_tile(1, 2, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        }
        silu_tile_init();
        silu_tile(0);
        load(CB_ACT, 0, 1);
        fill_tile_init();
        fill_tile(2, 0.0f);
        select(1, 0, 2, 0);
        tile_regs_commit();
        cq.reserve_back(1);
        tile_regs_wait();
        pack_reconfig_data_format(CB_QKV);
        pack_tile(0, CB_QKV);
        tile_regs_release();
        cq.push_back(1);

        // ---- history shift (h0' and h1')
        tile_regs_acquire();
        load(CB_ACT, 0, 0);
        load(CB_H, 1, 1);
        load(CB_H, 0, 2);
        select(0, 1, 2, 1);  // h0' -> DST 1
        load(CB_H, 2, 2);
        load(CB_H, 1, 3);
        select(0, 2, 3, 2);  // h1' -> DST 2
        tile_regs_commit();
        co.reserve_back(3);
        tile_regs_wait();
        pack_reconfig_data_format(CB_HOUT);
        pack_tile<true>(1, CB_HOUT, 0);
        pack_tile<true>(2, CB_HOUT, 1);
        tile_regs_release();

        // ---- h2'
        tile_regs_acquire();
        load(CB_ACT, 0, 0);
        load(CB_CUR, 0, 1);
        load(CB_H, 2, 2);
        select(0, 1, 2, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(1, CB_HOUT, 2);
        tile_regs_release();
        co.push_back(3);

        cc.pop_front(1);
        ch.pop_front(3);
        ct.pop_front(4);
    }
    ca.pop_front(1);
}
