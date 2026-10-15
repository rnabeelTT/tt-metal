// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused mHC pre-collapse on one core. Per (token-tile, column-tile) work unit:
//     h = sum_i pre_i * residual_i
// pre_i varies per token (tile row) but not per column, so each is first expanded into a full tile with
// `pre_tile @ consts[i]` (consts[i] has row i all ones: column i copied across the tile, exact), once per token-tile,
// and reused for every column tile. The n products accumulate in one DEST slot (mul_tiles with acc_to_dest).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

namespace {
constexpr uint32_t CB_IN = 0;
constexpr uint32_t CB_PRE = 1;
constexpr uint32_t CB_CONSTS = 2;
constexpr uint32_t CB_COEF = 3;
constexpr uint32_t CB_OUT = 16;
}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(CB_COEF, CB_IN, CB_OUT);
    const uint32_t num_units = get_arg_val<uint32_t>(0);
    const uint32_t start_unit = get_arg_val<uint32_t>(1);
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t col_tiles = get_compile_time_arg_val(1);
    if (num_units == 0) {
        return;
    }
    CircularBuffer cb_consts(CB_CONSTS), cb_pre(CB_PRE), cb_coef(CB_COEF), cb_in(CB_IN), cb_out(CB_OUT);
    cb_consts.wait_front(n);

    uint32_t cached_t0 = 0xFFFFFFFFu;
    for (uint32_t w = 0; w < num_units; ++w) {
        const uint32_t t0 = (start_unit + w) / col_tiles;
        if (t0 != cached_t0) {
            if (cached_t0 != 0xFFFFFFFFu) {
                cb_coef.pop_front(n);
            }
            cached_t0 = t0;
            cb_pre.wait_front(1);
            cb_coef.reserve_back(n);
            reconfig_data_format(CB_PRE, CB_CONSTS);
            pack_reconfig_data_format(CB_COEF);
            matmul_init(CB_PRE, CB_CONSTS);
            for (uint32_t k = 0; k < n; ++k) {
                tile_regs_acquire();
                matmul_tiles(CB_PRE, CB_CONSTS, 0, k, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, CB_COEF);
                tile_regs_release();
            }
            cb_coef.push_back(n);
            cb_pre.pop_front(1);
            cb_coef.wait_front(n);
        }

        cb_in.wait_front(n);
        cb_out.reserve_back(1);
        reconfig_data_format(CB_COEF, CB_IN);
        pack_reconfig_data_format(CB_OUT);
        tile_regs_acquire();
        mul_tiles_init(CB_COEF, CB_IN);
        for (uint32_t i = 0; i < n; ++i) {
            mul_tiles(CB_COEF, CB_IN, i, i, 0);  // += pre_i * residual_i (acc_to_dest)
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_OUT);
        tile_regs_release();
        cb_out.push_back(1);
        cb_in.pop_front(n);
    }
}
