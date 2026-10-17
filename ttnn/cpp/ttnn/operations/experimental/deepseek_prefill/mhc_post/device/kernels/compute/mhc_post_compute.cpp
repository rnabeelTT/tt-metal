// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused mHC post-mix on one core. Per (token-tile, column-tile) work unit:
//
//     out_j = post_j * y + sum_i comb_(i,j) * residual_i        for j in [0, n)
//
// post_j and comb_(i,j) vary per token (tile row) but not per column, so each is first expanded
// into a full tile: `coeff_tile @ consts[k]` copies column k across the whole tile, because
// consts[k] has row k all ones and is zero elsewhere. That is exact -- the products dropped are
// products with a true zero -- and it needs no sub-tile broadcast LLK, which cannot address an
// arbitrary column. The expansion is done once per token-tile and reused for every column tile.
//
// fp32 precision: the FPU reads srcA / srcB at TF32 (10-bit mantissa, truncating). So the raw post / comb tile is
// split x = hi + lo in DST (hi = x with the 13 low mantissa bits cleared: exactly TF32; lo = x - hi) and expanded as
// hi @ E + lo @ E (E is 0 / 1, exact); and the mix itself runs on the SFPU over DST tiles unpacked straight to DST
// (y, the streams and the coefficients are UnpackToDestFp32 CBs for an fp32 op, see the program factory):
// one multiply then n fused multiply-adds, all fp32.

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

constexpr uint32_t CB_IN = 0;      // c_0: y then the n residual-stream tiles
constexpr uint32_t CB_PC = 1;      // c_1: raw post tile, raw comb tile
constexpr uint32_t CB_CONSTS = 2;  // c_2: n*n column-broadcast tiles, resident
constexpr uint32_t CB_COEF = 3;    // c_3: n broadcast post_j then n*n broadcast comb_(i,j)
constexpr uint32_t CB_PCH = 4;     // c_4: post / comb hi
constexpr uint32_t CB_PCL = 5;     // c_5: post / comb lo
constexpr uint32_t CB_OUT = 16;

constexpr uint32_t ONE_BITS = 0x3f800000u;  // 1.0f

}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(CB_COEF, CB_IN, CB_OUT);  // must precede any other compute work
    const uint32_t num_units = get_arg_val<uint32_t>(0);
    const uint32_t start_unit = get_arg_val<uint32_t>(1);
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t col_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t num_coef = n + n * n;

    CircularBuffer cb_consts(CB_CONSTS), cb_pc(CB_PC), cb_coef(CB_COEF), cb_in(CB_IN), cb_out(CB_OUT);
    CircularBuffer cb_pch(CB_PCH), cb_pcl(CB_PCL);
    cb_consts.wait_front(n * n);

    uint32_t cached_t0 = 0xFFFFFFFFu;
    for (uint32_t w = 0; w < num_units; ++w) {
        const uint32_t t0 = (start_unit + w) / col_tiles;

        if (t0 != cached_t0) {
            if (cached_t0 != 0xFFFFFFFFu) {
                cb_coef.pop_front(num_coef);
            }
            cached_t0 = t0;

            // split the raw post / comb tiles into TF32-exact halves
            cb_pc.wait_front(2);
            cb_pch.reserve_back(2);
            cb_pcl.reserve_back(2);
            for (uint32_t s = 0; s < 2; ++s) {
                tile_regs_acquire();
                reconfig_data_format_srca(CB_PC);
                copy_init(CB_PC);
                copy_tile(CB_PC, s, 0);
                split_tile(0, 1, 2);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile<true>(1, CB_PCH, s);
                pack_tile<true>(2, CB_PCL, s);
                tile_regs_release();
            }
            cb_pch.push_back(2);
            cb_pcl.push_back(2);
            cb_pc.pop_front(2);
            cb_pch.wait_front(2);
            cb_pcl.wait_front(2);

            cb_coef.reserve_back(num_coef);
            reconfig_data_format(CB_PCH, CB_CONSTS);
            matmul_init(CB_PCH, CB_CONSTS);
            for (uint32_t k = 0; k < num_coef; ++k) {
                // post supplies the first n coefficients (its column j), comb the remaining n*n.
                const uint32_t src_tile = k < n ? 0 : 1;
                const uint32_t col = k < n ? k : k - n;
                tile_regs_acquire();
                matmul_tiles(CB_PCH, CB_CONSTS, src_tile, col, 0);
                matmul_tiles(CB_PCL, CB_CONSTS, src_tile, col, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, CB_COEF);
                tile_regs_release();
            }
            cb_coef.push_back(num_coef);
            cb_pch.pop_front(2);
            cb_pcl.pop_front(2);
            cb_coef.wait_front(num_coef);
        }

        cb_in.wait_front(1 + n);
        cb_out.reserve_back(n);
        for (uint32_t j = 0; j < n; ++j) {
            // CB_COEF and CB_IN share one data format / unpack mode, so one copy init serves both
            tile_regs_acquire();
            reconfig_data_format_srca(CB_IN);
            copy_init(CB_IN);
            copy_tile(CB_COEF, j, 1);
            copy_tile(CB_IN, 0, 2);
            mul_binary_tile_init();
            mul_binary_tile(1, 2, 0);  // post_j * y
            addcmul_tile_init();
            for (uint32_t i = 0; i < n; ++i) {
                copy_tile(CB_COEF, n + i * n + j, 1);
                copy_tile(CB_IN, 1 + i, 2);
                addcmul_tile<DataFormat::Float32>(0, 1, 2, 0, ONE_BITS);  // += comb_(i,j) * residual_i
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_OUT);
            tile_regs_release();
        }
        cb_out.push_back(n);
        cb_in.pop_front(1 + n);
    }
}
