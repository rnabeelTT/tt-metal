// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused mHC pre-collapse on one core. Per (token-tile, column-tile) work unit:
//     h = sum_i pre_i * residual_i
// pre_i varies per token (tile row) but not per column, so each is first expanded into a full tile with
// `pre_tile @ consts[i]` (consts[i] has row i all ones: column i copied across the tile, exact), once per token-tile,
// and reused for every column tile.
//
// fp32 precision: the FPU reads srcA / srcB at TF32 (10-bit mantissa, truncating). So the raw pre tile is split
// x = hi + lo in DST (hi = x with the 13 low mantissa bits cleared: exactly TF32; lo = x - hi) and expanded as
// hi @ E + lo @ E (E is 0 / 1, exact), and the products accumulate on the SFPU over DST tiles unpacked straight to DST
// (the streams and the coefficients are UnpackToDestFp32 CBs for an fp32 op, see the program factory).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

#ifdef TRISC_MATH
namespace ckernel::sfpu {
// DST[hi] = DST[x] with the 13 low mantissa bits cleared (TF32-exact), DST[lo] = DST[x] - DST[hi] (exact).
template <int ITERATIONS = 8>
inline void calculate_mhc_split_tf32(const std::uint32_t x, const std::uint32_t hi, const std::uint32_t lo) {
    constexpr std::uint32_t tile = 32;  // sfpi rows per DST tile
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[x * tile];
        sfpi::vFloat h = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(v) & 0xFFFFE000u);
        sfpi::dst_reg[hi * tile] = h;
        sfpi::dst_reg[lo * tile] = v - h;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

namespace {
ALWI void split_tile(uint32_t x, uint32_t hi, uint32_t lo) {
    MATH((SFPU_BINARY_INIT(unused)));
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_mhc_split_tf32, (8), x, hi, lo, VectorMode::RC)));
}

constexpr uint32_t CB_IN = 0;
constexpr uint32_t CB_PRE = 1;
constexpr uint32_t CB_CONSTS = 2;
constexpr uint32_t CB_COEF = 3;
constexpr uint32_t CB_PH = 4;  // pre hi
constexpr uint32_t CB_PL = 5;  // pre lo
constexpr uint32_t CB_OUT = 16;
constexpr uint32_t ONE_BITS = 0x3f800000u;  // 1.0f
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
    CircularBuffer cb_ph(CB_PH), cb_pl(CB_PL);
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
            cb_ph.reserve_back(1);
            cb_pl.reserve_back(1);
            pack_reconfig_data_format(CB_PH);
            tile_regs_acquire();
            reconfig_data_format_srca(CB_PRE);
            copy_init(CB_PRE);
            copy_tile(CB_PRE, 0, 0);
            split_tile(0, 1, 2);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(1, CB_PH);
            pack_tile(2, CB_PL);
            tile_regs_release();
            cb_ph.push_back(1);
            cb_pl.push_back(1);
            cb_pre.pop_front(1);
            cb_ph.wait_front(1);
            cb_pl.wait_front(1);

            cb_coef.reserve_back(n);
            reconfig_data_format(CB_PH, CB_CONSTS);
            matmul_init(CB_PH, CB_CONSTS);
            for (uint32_t k = 0; k < n; ++k) {
                tile_regs_acquire();
                matmul_tiles(CB_PH, CB_CONSTS, 0, k, 0);
                matmul_tiles(CB_PL, CB_CONSTS, 0, k, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, CB_COEF);
                tile_regs_release();
            }
            cb_coef.push_back(n);
            cb_ph.pop_front(1);
            cb_pl.pop_front(1);
            cb_coef.wait_front(n);
        }

        cb_in.wait_front(n);
        cb_out.reserve_back(1);
        pack_reconfig_data_format(CB_OUT);
        tile_regs_acquire();
        // CB_COEF and CB_IN share one data format / unpack mode, so one copy init serves both
        reconfig_data_format_srca(CB_IN);
        copy_init(CB_IN);
        copy_tile(CB_COEF, 0, 1);
        copy_tile(CB_IN, 0, 2);
        mul_binary_tile_init();
        mul_binary_tile(1, 2, 0);  // pre_0 * residual_0
        addcmul_tile_init();
        for (uint32_t i = 1; i < n; ++i) {
            copy_tile(CB_COEF, i, 1);
            copy_tile(CB_IN, i, 2);
            addcmul_tile<DataFormat::Float32>(0, 1, 2, 0, ONE_BITS);  // += pre_i * residual_i
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_OUT);
        tile_regs_release();
        cb_out.push_back(1);
        cb_in.pop_front(n);
    }
}
