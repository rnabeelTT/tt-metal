// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Message gate in one program (tt/gate.py), reader: this core's PER hidden tiles of h and m (fp32), its rows of the
// down projection Wd ([2H, R*32] bf16: the h rows then the m rows, tiles [t][n]), its columns of the forget / gain
// projections Wfg ([R*32, 2H] bf16: forget tiles [k][t], gain tiles [k][t]) and a bf16 ones tile (row sums).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t h_addr = get_arg_val<uint32_t>(0);
    const uint32_t m_addr = get_arg_val<uint32_t>(1);
    const uint32_t wd_addr = get_arg_val<uint32_t>(2);
    const uint32_t wfg_addr = get_arg_val<uint32_t>(3);
    const uint32_t t0 = get_arg_val<uint32_t>(4);  // this core's first hidden tile

    constexpr uint32_t cb_h = get_compile_time_arg_val(0);
    constexpr uint32_t cb_m = get_compile_time_arg_val(1);
    constexpr uint32_t cb_wd = get_compile_time_arg_val(2);
    constexpr uint32_t cb_wf = get_compile_time_arg_val(3);
    constexpr uint32_t cb_wg = get_compile_time_arg_val(4);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(5);
    constexpr uint32_t PER = get_compile_time_arg_val(6);
    constexpr uint32_t R = get_compile_time_arg_val(7);   // rank tiles
    constexpr uint32_t HT = get_compile_time_arg_val(8);  // hidden tiles
    constexpr uint32_t F32 = 4096, BF = 2048;
    constexpr auto x_args = TensorAccessorArgs<9>();
    constexpr auto wd_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto wfg_args = TensorAccessorArgs<wd_args.next_compile_time_args_offset()>();
    const auto ah = TensorAccessor(x_args, h_addr, F32);
    const auto am = TensorAccessor(x_args, m_addr, F32);
    const auto awd = TensorAccessor(wd_args, wd_addr, BF);
    const auto awfg = TensorAccessor(wfg_args, wfg_addr, BF);

    {
        cb_reserve_back(cb_ones, 1);
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ones));
        for (uint32_t i = 0; i < 512; ++i) {
            p[i] = 0x3F803F80u;
        }
        cb_push_back(cb_ones, 1);
    }
    cb_reserve_back(cb_h, PER);
    cb_reserve_back(cb_m, PER);
    for (uint32_t t = 0; t < PER; ++t) {
        noc_async_read(ah.get_noc_addr(t0 + t), get_write_ptr(cb_h) + t * F32, F32);
        noc_async_read(am.get_noc_addr(t0 + t), get_write_ptr(cb_m) + t * F32, F32);
    }
    noc_async_read_barrier();
    cb_push_back(cb_h, PER);
    cb_push_back(cb_m, PER);
    // Wd rows: tile (kt, n) is page kt * R + n; h rows kt = t0 + t, m rows kt = HT + t0 + t
    cb_reserve_back(cb_wd, 2 * PER * R);
    {
        const uint32_t w = get_write_ptr(cb_wd);
        for (uint32_t half = 0; half < 2; ++half) {
            for (uint32_t t = 0; t < PER; ++t) {
                for (uint32_t n = 0; n < R; ++n) {
                    noc_async_read(awd.get_noc_addr((half * HT + t0 + t) * R + n),
                                   w + ((half * PER + t) * R + n) * BF, BF);
                }
            }
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_wd, 2 * PER * R);
    // Wfg columns: tile (k, j) is page k * 2HT + j; forget j = t0 + t, gain j = HT + t0 + t
    cb_reserve_back(cb_wf, R * PER);
    cb_reserve_back(cb_wg, R * PER);
    for (uint32_t k = 0; k < R; ++k) {
        for (uint32_t t = 0; t < PER; ++t) {
            noc_async_read(awfg.get_noc_addr(k * 2 * HT + t0 + t), get_write_ptr(cb_wf) + (k * PER + t) * BF, BF);
            noc_async_read(awfg.get_noc_addr(k * 2 * HT + HT + t0 + t), get_write_ptr(cb_wg) + (k * PER + t) * BF, BF);
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_wf, R * PER);
    cb_push_back(cb_wg, R * PER);
}
