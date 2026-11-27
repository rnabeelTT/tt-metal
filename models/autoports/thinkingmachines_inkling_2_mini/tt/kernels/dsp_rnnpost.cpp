// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// DSpark Markov RNN step, the ops after the rnn_proj matmul in one program (tt/drafter.py _rnnpost); core b takes tile b
// of each part of pj = [sb | w1[prev] | h_k] @ rnn_proj ([1, 1, 1, 3R] FP32) and of cf:
//   lat = bf16(tanh(pj_2 + cf_2)) ; s' = lerp(tanh(pj_1 + cf_1), s, sigmoid(pj_0 + cf_0)) -> FP32 and bf16
// Replaces the cf add, the slices, sigmoid / tanh x2, the typecasts and the lerp. Three kernels from this file: READER
// (RISCV_1), WRITER (RISCV_0), else compute.
//
// runtime args (all): [pj, cf, s, lat, sn, sbn, b]   compile args: cb_in (7 FP32 tiles), cb_lat, cb_sn, cb_sb, RT

#include <stdint.h>

#if defined(READER) || defined(WRITER)
#include "api/dataflow/dataflow_api.h"
#else
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#endif

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_lat = get_compile_time_arg_val(1);
    constexpr uint32_t cb_sn = get_compile_time_arg_val(2);
    constexpr uint32_t cb_sb = get_compile_time_arg_val(3);
    constexpr uint32_t RT = get_compile_time_arg_val(4);
    const uint32_t b = get_arg_val<uint32_t>(6);
#if defined(READER) || defined(WRITER)
    constexpr auto a_args = TensorAccessorArgs<5>();
#endif
#if defined(READER)
    // cb_in: pj_0, pj_1, pj_2, cf_0, cf_1, cf_2, s
    const auto a_pj = TensorAccessor(a_args, get_arg_val<uint32_t>(0), 4096);
    const auto a_cf = TensorAccessor(a_args, get_arg_val<uint32_t>(1), 4096);
    const auto a_s = TensorAccessor(a_args, get_arg_val<uint32_t>(2), 4096);
    cb_reserve_back(cb_in, 7);
    const uint32_t l1 = get_write_ptr(cb_in);
    for (uint32_t c = 0; c < 3; ++c) {
        noc_async_read(a_pj.get_noc_addr(c * RT + b), l1 + c * 4096, 4096);
        noc_async_read(a_cf.get_noc_addr(c * RT + b), l1 + (3 + c) * 4096, 4096);
    }
    noc_async_read(a_s.get_noc_addr(b), l1 + 6 * 4096, 4096);
    noc_async_read_barrier();
    cb_push_back(cb_in, 7);
#elif defined(WRITER)
    cb_wait_front(cb_lat, 1);
    noc_async_write(get_read_ptr(cb_lat), TensorAccessor(a_args, get_arg_val<uint32_t>(3), 2048).get_noc_addr(b), 2048);
    cb_wait_front(cb_sn, 1);
    cb_wait_front(cb_sb, 1);
    noc_async_write(get_read_ptr(cb_sn), TensorAccessor(a_args, get_arg_val<uint32_t>(4), 4096).get_noc_addr(b), 4096);
    noc_async_write(get_read_ptr(cb_sb), TensorAccessor(a_args, get_arg_val<uint32_t>(5), 2048).get_noc_addr(b), 2048);
    noc_async_write_barrier();
    cb_pop_front(cb_lat, 1);
    cb_pop_front(cb_sn, 1);
    cb_pop_front(cb_sb, 1);
#else
    compute_kernel_hw_startup(cb_in, cb_lat);
    cb_wait_front(cb_in, 7);
    reconfig_data_format_srca(cb_in);
    copy_tile_to_dst_init_short(cb_in);
    tile_regs_acquire();
    copy_tile(cb_in, 2, 0);
    copy_tile(cb_in, 5, 1);
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    tanh_tile_init();
    tanh_tile(0);
    tile_regs_commit();
    cb_reserve_back(cb_lat, 1);
    pack_reconfig_data_format(cb_lat);
    tile_regs_wait();
    pack_tile(0, cb_lat);
    tile_regs_release();
    cb_push_back(cb_lat, 1);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_in);
    copy_tile(cb_in, 0, 0);
    copy_tile(cb_in, 1, 1);
    copy_tile(cb_in, 3, 2);
    copy_tile(cb_in, 4, 3);
    add_binary_tile_init();
    add_binary_tile(0, 2, 0);
    add_binary_tile(1, 3, 1);
    sigmoid_tile_init<false>();
    sigmoid_tile<VectorMode::RC, false>(0);
    tanh_tile_init();
    tanh_tile(1);
    copy_tile_to_dst_init_short(cb_in);
    copy_tile(cb_in, 6, 2);
    sub_binary_tile_init();
    sub_binary_tile(2, 1, 2);
    mul_binary_tile_init();
    mul_binary_tile(2, 0, 2);
    add_binary_tile_init();
    add_binary_tile(2, 1, 2);
    tile_regs_commit();
    cb_pop_front(cb_in, 7);
    cb_reserve_back(cb_sn, 1);
    cb_reserve_back(cb_sb, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_sn);
    pack_tile(2, cb_sn);
    pack_reconfig_data_format(cb_sb);
    pack_tile(2, cb_sb);
    tile_regs_release();
    cb_push_back(cb_sn, 1);
    cb_push_back(cb_sb, 1);
#endif
}
