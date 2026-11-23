// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Grouped expert matmul (tt/gmm.py), reader: this core's column block b of groups [g0, g1). Per group: its expert's
// [Kt x nb] weight block (re-read only when the expert changes: the single-block CB keeps the previous one in place),
// then the group's mt row tiles of A ([Kt] tiles each).
//
// runtime args: b, g0, g1      common args: a, w, ids

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_ids = get_compile_time_arg_val(2);
    constexpr uint32_t Kt = get_compile_time_arg_val(3);
    constexpr uint32_t Nt = get_compile_time_arg_val(4);
    constexpr uint32_t nb = get_compile_time_arg_val(5);
    constexpr uint32_t mt = get_compile_time_arg_val(6);
    constexpr uint32_t E = get_compile_time_arg_val(7);
    constexpr uint32_t ids_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t w_page = get_compile_time_arg_val(9);
    constexpr auto a_args = TensorAccessorArgs<10>();
    constexpr auto w_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto i_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const uint32_t b = get_arg_val<uint32_t>(0), g0 = get_arg_val<uint32_t>(1), g1 = get_arg_val<uint32_t>(2);
    const auto aa = TensorAccessor(a_args, get_common_arg_val<uint32_t>(0), 2048);
    const auto aw = TensorAccessor(w_args, get_common_arg_val<uint32_t>(1), w_page);
    const auto ai = TensorAccessor(i_args, get_common_arg_val<uint32_t>(2), ids_bytes);

    cb_reserve_back(cb_ids, 1);
    const uint32_t il = get_write_ptr(cb_ids);
    noc_async_read(ai.get_noc_addr(0), il, ids_bytes);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint16_t* ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(il);

    uint32_t prev = 0xFFFFFFFF;
    for (uint32_t g = g0; g < g1; ++g) {
        uint32_t e = ids[g];
        e = e < E ? e : 0;  // (padding groups: any expert; their rows are never read back)
        cb_reserve_back(cb_w, Kt * nb);
        if (e != prev) {
            const uint32_t wl = get_write_ptr(cb_w);
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                for (uint32_t j = 0; j < nb; ++j) {
                    noc_async_read(aw.get_noc_addr((e * Kt + kt) * Nt + b * nb + j), wl + (kt * nb + j) * w_page,
                                   w_page);
                }
            }
            noc_async_read_barrier();
            prev = e;
        }
        cb_push_back(cb_w, Kt * nb);
        for (uint32_t mi = 0; mi < mt; ++mi) {
            cb_reserve_back(cb_a, Kt);
            const uint32_t al = get_write_ptr(cb_a);
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                noc_async_read(aa.get_noc_addr((g * mt + mi) * Kt + kt), al + kt * 2048, 2048);
            }
            noc_async_read_barrier();
            cb_push_back(cb_a, Kt);
        }
    }
}
