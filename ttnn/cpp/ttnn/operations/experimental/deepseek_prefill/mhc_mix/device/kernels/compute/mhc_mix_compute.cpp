// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Per unit: DST 0 = sum_k x_k @ fn_k (projection partial), DST 1 = sum_k x_k @ x_k^T (Gram partial; its diagonal is the
// row sum of squares over the chunk). fp32 DST, HiFi4.
//
// fp32 precision: the FPU reads fp32 at TF32 (10-bit mantissa, truncating) and its multiplier uses only the top 10
// significant bits of srcA (matmul in1), which biases a sum of squares by ~1e-3. So each x tile (unpacked straight to
// DST: CB_X is UnpackToDestFp32) is split x = hi + lo (hi = 10 significant bits, used whole by either source; lo =
// x - hi, exact) and the products are taken as
//   x @ fn = hi @ fn + lo @ fn,     x @ x^T = hi @ hi^T + hi @ lo^T + lo @ hi^T + lo @ lo^T     (~1e-6 relative).
// fn is used at srcA precision (exact for the bf16-valued mHC projection weights).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"

#ifdef TRISC_MATH
namespace ckernel::sfpu {
// DST[hi] = DST[x] with the 14 low mantissa bits cleared (10 significant bits), DST[lo] = DST[x] - DST[hi] (exact).
template <int ITERATIONS = 8>
inline void calculate_mhc_split_tf32(const std::uint32_t x, const std::uint32_t hi, const std::uint32_t lo) {
    constexpr std::uint32_t tile = 32;  // sfpi rows per DST tile
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[x * tile];
        sfpi::vFloat h = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(v) & 0xFFFFC000u);
        sfpi::dst_reg[hi * tile] = h;
        sfpi::dst_reg[lo * tile] = v - h;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

void kernel_main() {
    constexpr uint32_t Kc = get_compile_time_arg_val(0);
    const uint32_t num_units = get_arg_val<uint32_t>(0);
    constexpr uint32_t CB_X = 0, CB_FN = 1, CB_XH = 2, CB_XL = 3, CB_OUT = 16;
    if (num_units == 0) {
        return;
    }
    compute_kernel_hw_startup(CB_XH, CB_FN, CB_OUT);
    CircularBuffer cx(CB_X), cfn(CB_FN), cxh(CB_XH), cxl(CB_XL), co(CB_OUT);
    for (uint32_t w = 0; w < num_units; ++w) {
        cx.wait_front(Kc);
        cxh.reserve_back(Kc);
        cxl.reserve_back(Kc);
        for (uint32_t k = 0; k < Kc; ++k) {
            tile_regs_acquire();
            reconfig_data_format_srca(CB_X);
            copy_init(CB_X);
            copy_tile(CB_X, k, 0);
            MATH((SFPU_BINARY_INIT(unused)));
            MATH((SFPU_BINARY_CALL(
                DST_SYNC_MODE, DST_ACCUM_MODE, calculate_mhc_split_tf32, (8), 0, 1, 2, VectorMode::RC)));
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(1, CB_XH, k);
            pack_tile<true>(2, CB_XL, k);
            tile_regs_release();
        }
        cxh.push_back(Kc);
        cxl.push_back(Kc);
        cx.pop_front(Kc);

        cxh.wait_front(Kc);
        cxl.wait_front(Kc);
        cfn.wait_front(Kc);
        co.reserve_back(2);
        tile_regs_acquire();
        reconfig_data_format(CB_XH, CB_FN);
        matmul_init(CB_XH, CB_FN, 0);
        for (uint32_t k = 0; k < Kc; ++k) {
            matmul_tiles(CB_XH, CB_FN, k, k, 0);
            matmul_tiles(CB_XL, CB_FN, k, k, 0);
        }
        matmul_init(CB_XH, CB_XH, 1);
        for (uint32_t k = 0; k < Kc; ++k) {
            matmul_tiles(CB_XH, CB_XH, k, k, 1);
            matmul_tiles(CB_XH, CB_XL, k, k, 1);
            matmul_tiles(CB_XL, CB_XH, k, k, 1);
            matmul_tiles(CB_XL, CB_XL, k, k, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_OUT);
        pack_tile(1, CB_OUT);
        tile_regs_release();
        co.push_back(2);
        cxh.pop_front(Kc);
        cxl.pop_front(Kc);
        cfn.pop_front(Kc);
    }
}
