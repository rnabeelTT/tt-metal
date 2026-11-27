// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Prefill attention prep in the projections' own layout (tt/aprep.py NAT), reader. The chunk's rows stay in their tile
// rows: unit (R, kind) = tile row R (rows 32 R .. 32 R + 31) x one of: a q head group, an r head group, a k head, a v
// head, the sink tile (aprep_nat.hpp). Per head the reader pushes whole tiles: q head h = pq tiles R PQT + h HT + c
// (fp32), k / v head = tiles R KVT + h HT + c (bf16), r head h = pq tile R0 + h / 2 with head h's 16 columns moved to
// columns 0..15 and columns 16..31 zeroed (DR 16; DR 32: tile R0 + h as is), the sink = pq tile S0. q / r units of a
// global layer first get a tau tile: row i = the log scaling of row 32 R + i in every column (fp32).
// Every row is a token: the compute's per-row math is the per-row prep's (aprep_compute.cpp) on whole tiles.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "aprep_nat.hpp"

void kernel_main() {
    const uint32_t pq_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t tau_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t RB0 = get_common_arg_val<uint32_t>(4);  // (pq / k / v: this call's rows from row tile RB0 on)
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t step = get_arg_val<uint32_t>(1);
    const uint32_t total = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_xf = get_compile_time_arg_val(0);
    constexpr uint32_t cb_xb = get_compile_time_arg_val(1);
    constexpr uint32_t cb_tau = get_compile_time_arg_val(2);
    constexpr uint32_t HT = get_compile_time_arg_val(3);
    constexpr uint32_t DR = get_compile_time_arg_val(4);
    constexpr uint32_t R0 = get_compile_time_arg_val(5);
    constexpr uint32_t S0 = get_compile_time_arg_val(6);
    constexpr uint32_t GLOBAL = get_compile_time_arg_val(7);
    constexpr uint32_t PQT = get_compile_time_arg_val(8);
    constexpr uint32_t KVT = get_compile_time_arg_val(9);
    constexpr uint32_t cb_x1 = get_compile_time_arg_val(10);  // (1-tile pushes: r heads, the sink)
    constexpr auto pq_args = TensorAccessorArgs<11>();
    constexpr auto kv_args = TensorAccessorArgs<pq_args.next_compile_time_args_offset()>();
    constexpr auto tau_args = TensorAccessorArgs<kv_args.next_compile_time_args_offset()>();
    constexpr uint32_t F32_TILE = 4096, BF_TILE = 2048;
    const auto apq = TensorAccessor(pq_args, pq_addr, F32_TILE);
    const auto ak = TensorAccessor(kv_args, k_addr, BF_TILE);
    const auto av = TensorAccessor(kv_args, v_addr, BF_TILE);
    const auto atau = TensorAccessor(tau_args, tau_addr, F32_TILE);

    for (uint32_t u = start; u < total; u += step) {
        const uint32_t R = u / NAT_NU;
        uint32_t type, h0, h1;
        nat_unit(u % NAT_NU, type, h0, h1);
        if (GLOBAL && type <= 1) {  // the rows' tau, row i in every column
            cb_reserve_back(cb_tau, 1);
            const uint32_t l1 = get_write_ptr(cb_tau);
            for (uint32_t i = 0; i < 32; ++i) {  // (element 0 of row b's tau tile -> word 0 of row i's first face row)
                noc_async_read(atau.get_noc_addr(R * 32 + i, 0), l1 + nat_f32_row(i, 0), 64);
            }
            noc_async_read_barrier();
            volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
            for (uint32_t i = 0; i < 32; ++i) {
                const uint32_t o0 = nat_f32_row(i, 0) / 4, o1 = nat_f32_row(i, 1) / 4;
                const uint32_t t = p[o0];
                for (uint32_t w = 0; w < 16; ++w) {
                    p[o0 + w] = t;
                    p[o1 + w] = t;
                }
            }
            cb_push_back(cb_tau, 1);
        }
        if (type == 0) {
            for (uint32_t h = h0; h < h1; ++h) {
                cb_reserve_back(cb_xf, HT);
                const uint32_t l1 = get_write_ptr(cb_xf);
                for (uint32_t c = 0; c < HT; ++c) {
                    noc_async_read(apq.get_noc_addr((R + RB0) * PQT + h * HT + c), l1 + c * F32_TILE, F32_TILE);
                }
                noc_async_read_barrier();
                cb_push_back(cb_xf, HT);
            }
        } else if (type == 1) {
            for (uint32_t h = h0; h < h1; ++h) {
                cb_reserve_back(cb_x1, 1);
                const uint32_t l1 = get_write_ptr(cb_x1);
                noc_async_read(apq.get_noc_addr((R + RB0) * PQT + R0 + (DR == 16 ? h / 2 : h)), l1, F32_TILE);
                noc_async_read_barrier();
                if constexpr (DR == 16) {
                    // fp32 faces of 1 KB: 0 = rows 0-15 cols 0-15, 1 = rows 0-15 cols 16-31, 2 / 3 rows 16-31
                    invalidate_l1_cache();
                    uint32_t* p = reinterpret_cast<uint32_t*>(l1);
                    if (h & 1) {
                        for (uint32_t w = 0; w < 256; ++w) {
                            p[w] = p[256 + w];
                            p[512 + w] = p[768 + w];
                        }
                    }
                    for (uint32_t w = 0; w < 256; ++w) {
                        p[256 + w] = 0;
                        p[768 + w] = 0;
                    }
                }
                cb_push_back(cb_x1, 1);
            }
        } else if (type == 2 || type == 3) {
            const auto& a = type == 2 ? ak : av;
            cb_reserve_back(cb_xb, HT);
            const uint32_t l1 = get_write_ptr(cb_xb);
            for (uint32_t c = 0; c < HT; ++c) {
                noc_async_read(a.get_noc_addr((R + RB0) * KVT + h0 * HT + c), l1 + c * BF_TILE, BF_TILE);
            }
            noc_async_read_barrier();
            cb_push_back(cb_xb, HT);
        } else {
            cb_reserve_back(cb_x1, 1);
            noc_async_read(apq.get_noc_addr((R + RB0) * PQT + S0), get_write_ptr(cb_x1), F32_TILE);
            noc_async_read_barrier();
            cb_push_back(cb_x1, 1);
        }
    }
}
