// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// DSpark drafter MLP activation (tt/drafter.py _swiglu): gu [1, 1, R <= 32, 2 I] bf16 TILE (one tile row) ->
// act [1, 1, R, I] bf16 = silu(g) * u, tile t of act from tiles t (g) and t + IT (u) of gu. Replaces the two slices and
// the multiply. Three kernels from this file: READER (RISCV_1), WRITER (RISCV_0), else compute.
//
// runtime args (all): [gu_addr, act_addr, t0, nt]

#include <stdint.h>

#if defined(READER) || defined(WRITER)
#include "api/dataflow/dataflow_api.h"
#else
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#endif

constexpr uint32_t TB = 2048;

void kernel_main() {
    const uint32_t gu_addr = get_arg_val<uint32_t>(0);
    const uint32_t act_addr = get_arg_val<uint32_t>(1);
    const uint32_t t0 = get_arg_val<uint32_t>(2);
    const uint32_t nt = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_g = get_compile_time_arg_val(0);
    constexpr uint32_t cb_u = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t IT = get_compile_time_arg_val(3);
#if defined(READER)
    constexpr auto g_args = TensorAccessorArgs<4>();
    const auto s = TensorAccessor(g_args, gu_addr, TB);
    for (uint32_t i = 0; i < nt; ++i) {
        cb_reserve_back(cb_g, 1);
        cb_reserve_back(cb_u, 1);
        noc_async_read(s.get_noc_addr(t0 + i), get_write_ptr(cb_g), TB);
        noc_async_read(s.get_noc_addr(t0 + i + IT), get_write_ptr(cb_u), TB);
        noc_async_read_barrier();
        cb_push_back(cb_g, 1);
        cb_push_back(cb_u, 1);
    }
#elif defined(WRITER)
    constexpr auto g_args = TensorAccessorArgs<4>();
    constexpr auto o_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    const auto d = TensorAccessor(o_args, act_addr, TB);
    for (uint32_t i = 0; i < nt; ++i) {
        cb_wait_front(cb_out, 1);
        noc_async_write(get_read_ptr(cb_out), d.get_noc_addr(t0 + i), TB);
        noc_async_writes_flushed();
        cb_pop_front(cb_out, 1);
    }
    noc_async_write_barrier();
#else
    if (nt == 0) {
        return;
    }
    compute_kernel_hw_startup(cb_g, cb_out);
    for (uint32_t i = 0; i < nt; ++i) {
        cb_wait_front(cb_g, 1);
        cb_wait_front(cb_u, 1);
        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_g);
        copy_tile(cb_g, 0, 0);
        silu_tile_init();
        silu_tile(0);
        copy_tile_to_dst_init_short(cb_u);
        copy_tile(cb_u, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        cb_pop_front(cb_g, 1);
        cb_pop_front(cb_u, 1);
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
#endif
}
