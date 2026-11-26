// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Prefill attention output gate (tt/agate.py PrefillGate), compute: tt/kernels/agate_compute.cpp's math on whole tiles:
// out = bf16(o sigmoid(g)), fp32, rounded to nearest even on the SFPU (the typecast's).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"


#ifdef TRISC_MATH
namespace ckernel::sfpu {
template <int ITERATIONS>
inline void calculate_pg_mul_rbf16(const uint32_t a, const uint32_t b, const uint32_t o) {
    constexpr uint32_t tile = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[a * tile];
        sfpi::vFloat y = sfpi::dst_reg[b * tile];
        sfpi::vFloat p = x * y;
        sfpi::vUInt u = sfpi::as<sfpi::vUInt>(p);
        sfpi::vUInt lsb = (u >> 16) & 1;
        u = (u + 0x7FFF + lsb) & 0xFFFF0000;
        sfpi::dst_reg[o * tile] = sfpi::as<sfpi::vFloat>(u);
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

void kernel_main() {
    constexpr uint32_t cb_o = get_compile_time_arg_val(0);
    constexpr uint32_t cb_g = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    const uint32_t n = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_g, cb_out);
    for (uint32_t jj = 0; jj < n; ++jj) {
        cb_wait_front(cb_g, 1);
        cb_wait_front(cb_o, 1);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_g);
        copy_init(cb_g);
        copy_tile(cb_g, 0, 0);
        reconfig_data_format_srca(cb_o);
        copy_init(cb_o);
        copy_tile(cb_o, 0, 1);
        sigmoid_tile_init<false>();
        sigmoid_tile<VectorMode::RC, false>(0);
#ifdef TRISC_MATH
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
#endif
        MATH((SFPU_BINARY_INIT(unused)));
        MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_pg_mul_rbf16, (8), 0, 1, 0, VectorMode::RC)));
        tile_regs_commit();
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_out);
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_g, 1);
        cb_pop_front(cb_o, 1);
    }
}
