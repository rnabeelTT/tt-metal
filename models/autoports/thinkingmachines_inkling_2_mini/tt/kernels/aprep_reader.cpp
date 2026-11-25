// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Decode attention prep (tt/aprep.py), reader. Unit u = b * 5 + type (user b; type 0 q, 1 r, 2 k, 3 v, 4 sink), units
// start, start + step, ... < total on this core. Head-major gather: output tile c of a unit with `heads` heads of `nt`
// tiles each holds in row h the source row b of source tile (off + h * nt + c) (two face-row reads per head); rows past
// `heads` are zero. q / r / sink come from the fp32 projection output pq ([1, 1, B, fq], one tile row), k / v from the
// bf16 conv outputs ([1, 1, B, kd]). The sink unit's tile holds s_h in every column of row h. q / r units of a global
// layer also get user b's tau tile (fp32, element 0).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "aprep_units.hpp"

namespace {
constexpr uint32_t F32_FACE = 1024, F32_ROW = 64, F32_TILE = 4096;
constexpr uint32_t BF_FACE = 512, BF_ROW = 32, BF_TILE = 2048;

inline uint32_t row_off(uint32_t r, uint32_t half, uint32_t face, uint32_t row) {
    return ((r >> 4) * 2 + half) * face + (r & 15) * row;
}
inline uint32_t f32_idx(uint32_t r, uint32_t col) {
    return (((r >> 4) * 2 + (col >> 4)) * 256) + (r & 15) * 16 + (col & 15);
}
inline void zero(uint32_t l1, uint32_t bytes) {
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t o = 0; o < bytes; o += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, l1 + o, bytes - o < MEM_ZEROS_SIZE ? bytes - o : MEM_ZEROS_SIZE);
    }
}
#ifndef APREP_PQT
#define APREP_PQT 0  // (tiles per row of pq / k / v: prefill rows past the first tile row; decode B <= 32 needs none)
#define APREP_KVT 0
#endif
// row b of src tiles (off + h * nt + c), h < heads, into row h of nt zeroed tiles at l1 (row b: row b % 32 of tile row
// b / 32, rowt tiles per row)
template <typename Acc>
inline void gather(const Acc& src, uint32_t l1, uint32_t off, uint32_t heads, uint32_t nt, uint32_t b, uint32_t face,
                   uint32_t row, uint32_t tile, uint32_t rowt, uint32_t h0 = 0, bool zf = true, uint32_t rb = 0) {
    if (zf) {
        zero(l1, nt * tile);
        noc_async_read_barrier();
    }
    const uint32_t base = (b >> 5) * rowt, br = b & 31;
    for (uint32_t h = h0; h < heads; ++h) {
        for (uint32_t c = 0; c < nt; ++c) {
            for (uint32_t half = 0; half < 2; ++half) {
                noc_async_read(src.get_noc_addr(base + off + h * nt + c, row_off(br, half, face, row)),
                               l1 + c * tile + row_off(h - rb, half, face, row), row);
            }
        }
    }
}
}  // namespace

#ifndef APREP_ONES_C
#define APREP_ONES_C 0
#endif
#ifndef APREP_RB
#define APREP_RB 0  // (> 0: head group rows rebased to row 0, see aprep_compute.cpp)
#endif
// APREP_NOZERO: a unit whose writer stores only its own head rows (q / k / v groups, decode) skips zeroing the
// other rows (the math is row-local: they never reach an output)
#ifndef APREP_NOZERO
#define APREP_NOZERO 0
#endif
constexpr bool ZQ = !(APREP_NOZERO && APREP_PQT == 0 && APREP_QN > 1);
constexpr bool ZKV = !(APREP_NOZERO && APREP_PQT == 0 && (APREP_KN > 1 || APREP_KVW));

void kernel_main() {
#ifdef APREP_CARGS  // (the addresses in the common runtime args; per core: start, step, total)
    const uint32_t pq_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t tau_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t step = get_arg_val<uint32_t>(1);
    const uint32_t total = get_arg_val<uint32_t>(2);
#else
    const uint32_t pq_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t tau_addr = get_arg_val<uint32_t>(3);
    const uint32_t start = get_arg_val<uint32_t>(4);
    const uint32_t step = get_arg_val<uint32_t>(5);
    const uint32_t total = get_arg_val<uint32_t>(6);
#endif

    constexpr uint32_t cb_xf = get_compile_time_arg_val(0);
    constexpr uint32_t cb_xb = get_compile_time_arg_val(1);
    constexpr uint32_t cb_tau = get_compile_time_arg_val(2);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(3);
    constexpr uint32_t NH = get_compile_time_arg_val(4);
    constexpr uint32_t NKV = get_compile_time_arg_val(5);
    constexpr uint32_t HT = get_compile_time_arg_val(6);   // head_dim tiles (4)
    constexpr uint32_t DR = get_compile_time_arg_val(7);   // rel dim: 16 (two heads per pq tile) or 32
    constexpr uint32_t R0 = get_compile_time_arg_val(8);   // first r tile of pq
    constexpr uint32_t S0 = get_compile_time_arg_val(9);   // the sink tile of pq
    constexpr uint32_t GLOBAL = get_compile_time_arg_val(10);
    static_assert(NH < 32, "the sink unit stages its row in tile row 31");
    constexpr auto pq_args = TensorAccessorArgs<11>();
    constexpr auto kv_args = TensorAccessorArgs<pq_args.next_compile_time_args_offset()>();
    constexpr auto tau_args = TensorAccessorArgs<kv_args.next_compile_time_args_offset()>();
    const auto apq = TensorAccessor(pq_args, pq_addr, F32_TILE);
    const auto ak = TensorAccessor(kv_args, k_addr, BF_TILE);
    const auto av = TensorAccessor(kv_args, v_addr, BF_TILE);
    const auto atau = TensorAccessor(tau_args, tau_addr, F32_TILE);

    // a bf16 tile of ones (the row-sum matmul operand), once (APREP_ONES_C: made by the compute, off the read path)
    if constexpr (!APREP_ONES_C) {
        cb_reserve_back(cb_ones, 1);
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ones));
        for (uint32_t i = 0; i < 512; ++i) {
            p[i] = 0x3F803F80u;
        }
        cb_push_back(cb_ones, 1);
    }
    for (uint32_t u = start; u < total; u += step) {
        const uint32_t b = u / APREP_NU;
        uint32_t type, h0, h1;
        aprep_unit(u % APREP_NU, NH, NKV, type, h0, h1);
        if (type == 0 || type == 1) {
            const uint32_t nt = type == 0 ? HT : 1;
            cb_reserve_back(cb_xf, nt);
            if (type == 0) {
                gather(apq, get_write_ptr(cb_xf), 0, h1, nt, b, F32_FACE, F32_ROW, F32_TILE, APREP_PQT, h0, ZQ, APREP_RB ? h0 : 0);
            } else if (DR == 32) {
                gather(apq, get_write_ptr(cb_xf), R0, NH, nt, b, F32_FACE, F32_ROW, F32_TILE, APREP_PQT);
            } else {  // DR 16: head h = face half h % 2 of pq tile R0 + h / 2 -> columns 0..15 of row h (16..31 zero)
                const uint32_t l1 = get_write_ptr(cb_xf);
                zero(l1, F32_TILE);
                noc_async_read_barrier();
                for (uint32_t h = 0; h < NH; ++h) {
                    noc_async_read(apq.get_noc_addr((b >> 5) * APREP_PQT + R0 + h / 2,
                                                    row_off(b & 31, h % 2, F32_FACE, F32_ROW)),
                                   l1 + row_off(h, 0, F32_FACE, F32_ROW), F32_ROW);
                }
            }
            if constexpr (GLOBAL) {
                cb_reserve_back(cb_tau, 1);
                noc_async_read(atau.get_noc_addr(b, 0), get_write_ptr(cb_tau), F32_ROW);  // element (0, 0) at word 0
            }
            noc_async_read_barrier();
            cb_push_back(cb_xf, nt);
            if constexpr (GLOBAL) {
                cb_push_back(cb_tau, 1);
            }
        } else if (type == 2 || type == 3) {
            cb_reserve_back(cb_xb, HT);
            gather(type == 2 ? ak : av, get_write_ptr(cb_xb), 0, h1, HT, b, BF_FACE, BF_ROW, BF_TILE, APREP_KVT, h0, ZKV, APREP_RB ? h0 : 0);
            noc_async_read_barrier();
            cb_push_back(cb_xb, HT);
        } else {
            // sink: row b of the sink tile -> s_h in every column of row h
            cb_reserve_back(cb_xf, 1);
            const uint32_t l1 = get_write_ptr(cb_xf);
            zero(l1, F32_TILE);
            noc_async_read_barrier();
            // stage row b (32 values) in row 31 of the tile (rows >= NH are zeroed again below)
            for (uint32_t half = 0; half < 2; ++half) {
                noc_async_read(apq.get_noc_addr((b >> 5) * APREP_PQT + S0, row_off(b & 31, half, F32_FACE, F32_ROW)),
                               l1 + row_off(31, half, F32_FACE, F32_ROW), F32_ROW);
            }
            noc_async_read_barrier();
            volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
            uint32_t s[NH];
            for (uint32_t h = 0; h < NH; ++h) {
                s[h] = p[f32_idx(31, h)];
            }
            for (uint32_t col = 0; col < 32; ++col) {
                p[f32_idx(31, col)] = 0;
            }
            for (uint32_t h = 0; h < NH; ++h) {  // (row h: 16 words in each of its two faces)
                volatile tt_l1_ptr uint32_t* r = p + (h >> 4) * 512 + (h & 15) * 16;
                const uint32_t v = s[h];
                for (uint32_t col = 0; col < 16; ++col) {
                    r[col] = v;
                    r[256 + col] = v;
                }
            }
            cb_push_back(cb_xf, 1);
        }
    }
}
