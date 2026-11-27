// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// DSpark drafter attention prep in one program (tt/drafter.py _qkv): from a projection output [1, 1, R <= 32, C] bf16
// TILE (one tile row; heads = consecutive 128-column groups) per unit (one per core):
//   kind 0 (q head g G + j): RMSNorm (gamma) + neox RoPE -> rows j R .. j R + R-1 of item g of the SDPA query
//          [1, NKV, G R, 128] (one tile row per item), assembled in the group leader's (j = 0) L1, written once
//   kind 1 (k head): RMSNorm + RoPE -> cache rows (p0 + i) mod S of head h ([NKV S / 32, 1, 32, 128] paged order)
//   kind 2 (v head): copied -> cache rows (p0 + i) mod S
// Replaces the head split, the per-head rms_norm / rotary_embedding ops, the KV cache write and the query reshape.
// Arithmetic: the norm as ttnn.rms_norm (fp32 DST: squares, row sums, x 1/W, + eps, rsqrt, x r, x gamma -> bf16);
// RoPE as rotary_embedding (bf16 x C -> bf16, bf16 rot(x) x S' -> bf16, sum -> bf16; S' = [-s, s] carries the sign).
// Three kernels from this file: READER (RISCV_1), WRITER (RISCV_0), else compute.
//
// runtime args (all): [src, kind, nh, c0, gamma, C table, S' table, dst, idx, Srows, p0_addr, j, leader x, leader y,
// G]; the RoPE rows are p0 + i (block: p0 = the block's first position; commit: the first committed row)

#include <stdint.h>

#if defined(READER) || defined(WRITER)
#include "api/dataflow/dataflow_api.h"
#else
#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#endif

constexpr uint32_t TB = 2048;
constexpr uint32_t DT = 4;  // tiles per head (head_dim 128)

constexpr uint32_t cb_x = get_compile_time_arg_val(0);
constexpr uint32_t cb_gm = get_compile_time_arg_val(1);
constexpr uint32_t cb_c = get_compile_time_arg_val(2);
constexpr uint32_t cb_s = get_compile_time_arg_val(3);
constexpr uint32_t cb_scaler = get_compile_time_arg_val(4);
constexpr uint32_t cb_eps = get_compile_time_arg_val(5);
constexpr uint32_t cb_sq = get_compile_time_arg_val(6);
constexpr uint32_t cb_ex2 = get_compile_time_arg_val(7);
constexpr uint32_t cb_r = get_compile_time_arg_val(8);
constexpr uint32_t cb_fus = get_compile_time_arg_val(9);
constexpr uint32_t cb_xn = get_compile_time_arg_val(10);
constexpr uint32_t cb_cm = get_compile_time_arg_val(11);
constexpr uint32_t cb_sm = get_compile_time_arg_val(12);
constexpr uint32_t cb_out = get_compile_time_arg_val(13);
constexpr uint32_t cb_wr = get_compile_time_arg_val(14);  // writer scratch: DT tiles + 1
constexpr uint32_t R = get_compile_time_arg_val(15);
constexpr uint32_t eps_bits = get_compile_time_arg_val(16);
constexpr uint32_t inv_w = get_compile_time_arg_val(17);

#if defined(READER) || defined(WRITER)
static inline void copy_row(uint32_t src_tile, uint32_t sr, uint32_t dst_tile, uint32_t dr) {
    // a tile row = 16 bf16 in face (r / 16) * 2 and 16 in the next face
    for (uint32_t f = 0; f < 2; ++f) {
        volatile tt_l1_ptr uint32_t* s =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_tile + ((sr >> 4) * 2 + f) * 512 + (sr & 15) * 32);
        volatile tt_l1_ptr uint32_t* d =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_tile + ((dr >> 4) * 2 + f) * 512 + (dr & 15) * 32);
        for (uint32_t w = 0; w < 8; ++w) {
            d[w] = s[w];
        }
    }
}
#endif

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t kind = get_arg_val<uint32_t>(1);
    const uint32_t nh = get_arg_val<uint32_t>(2);
#if defined(READER)
    const uint32_t c0 = get_arg_val<uint32_t>(3);
    const uint32_t g_addr = get_arg_val<uint32_t>(4);
    const uint32_t c_addr = get_arg_val<uint32_t>(5);
    const uint32_t s_addr = get_arg_val<uint32_t>(6);
    constexpr auto a_args = TensorAccessorArgs<18>();
    const auto as = TensorAccessor(a_args, src_addr, TB);
    if (kind != 2) {
        // RoPE rows p0 + i of the row-major C / S' tables ([rows, 128] bf16, 256 B rows) -> rows i of the tiles
        constexpr auto p_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
        const auto ag = TensorAccessor(a_args, g_addr, TB);
        const auto ac = TensorAccessor(a_args, c_addr, 256);
        const auto asn = TensorAccessor(a_args, s_addr, 256);
        cb_reserve_back(cb_gm, DT);
        cb_reserve_back(cb_c, DT);
        cb_reserve_back(cb_s, DT);
        const uint32_t stage = get_write_ptr(cb_sm);  // (free until the compute's RoPE: 2 R rows of 256 B fit)
        noc_async_read(TensorAccessor(p_args, get_arg_val<uint32_t>(10), 4).get_noc_addr(0), stage, 4);
        for (uint32_t d = 0; d < DT; ++d) {
            noc_async_read(ag.get_noc_addr(d), get_write_ptr(cb_gm) + d * TB, TB);
        }
        noc_async_read_barrier();
        const uint32_t p0 = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage);
        for (uint32_t i = 0; i < R; ++i) {
            noc_async_read(ac.get_noc_addr(p0 + i), stage + i * 256, 256);
            noc_async_read(asn.get_noc_addr(p0 + i), stage + (R + i) * 256, 256);
        }
        noc_async_read_barrier();
        for (uint32_t t = 0; t < 2; ++t) {
            const uint32_t dst = get_write_ptr(t ? cb_s : cb_c);
            for (uint32_t i = 0; i < R; ++i) {
                volatile tt_l1_ptr uint32_t* src =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage + (t * R + i) * 256);
                for (uint32_t d = 0; d < DT; ++d) {
                    for (uint32_t f = 0; f < 2; ++f) {
                        volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                            dst + d * TB + ((i >> 4) * 2 + f) * 512 + (i & 15) * 32);
                        for (uint32_t w = 0; w < 8; ++w) {
                            o[w] = src[d * 16 + f * 8 + w];
                        }
                    }
                }
            }
        }
        // scaler: 1.0 (bf16) in row 0 of each face; eps: bf16 in column 0 of faces 0 / 2
        cb_reserve_back(cb_scaler, 1);
        {
            volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_scaler));
            for (uint32_t i = 0; i < 512; ++i) {
                p[i] = 0;
            }
            for (uint32_t f = 0; f < 4; ++f) {
                for (uint32_t j = 0; j < 8; ++j) {
                    p[f * 128 + j] = 0x3f803f80u;
                }
            }
        }
        cb_push_back(cb_scaler, 1);
        cb_reserve_back(cb_eps, 1);
        {
            volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_eps));
            for (uint32_t i = 0; i < 1024; ++i) {
                p[i] = 0;
            }
            for (uint32_t k = 0; k < 4; k += 2) {
                for (uint32_t j = 0; j < 256; j += 16) {
                    p[(k << 8) + j] = static_cast<uint16_t>(eps_bits >> 16);
                }
            }
        }
        cb_push_back(cb_eps, 1);
        noc_async_read_barrier();
        cb_push_back(cb_gm, DT);
        cb_push_back(cb_c, DT);
        cb_push_back(cb_s, DT);
    }
    for (uint32_t h = 0; h < nh; ++h) {
        cb_reserve_back(cb_x, DT);
        const uint32_t l1 = get_write_ptr(cb_x);
        for (uint32_t d = 0; d < DT; ++d) {
            noc_async_read(as.get_noc_addr(c0 + h * DT + d), l1 + d * TB, TB);
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, DT);
    }
#elif defined(WRITER)
    const uint32_t dst_addr = get_arg_val<uint32_t>(7);
    const uint32_t idx = get_arg_val<uint32_t>(8);
    const uint32_t S = get_arg_val<uint32_t>(9);
    const uint32_t p0_addr = get_arg_val<uint32_t>(10);
    constexpr auto a_args = TensorAccessorArgs<18>();
    constexpr auto p_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto ad = TensorAccessor(a_args, dst_addr, TB);
    const uint32_t wr = get_write_ptr(cb_wr), scratch = wr + DT * TB;
    if (kind == 0) {
        // one q head per core: its R rows -> rows j R .. of the group leader's staging tiles (L1 -> L1, 32 B row
        // halves), a semaphore increment there; the leader (j = 0) zeroes the rows past G R, waits for all G heads and
        // writes the item's tiles
        const uint32_t j = get_arg_val<uint32_t>(11);
        const uint32_t lx = get_arg_val<uint32_t>(12), ly = get_arg_val<uint32_t>(13);
        const uint32_t G = get_arg_val<uint32_t>(14);
        const uint32_t sem = get_semaphore(0);
        if (j == 0) {
            for (uint32_t d = 0; d < DT; ++d) {
                for (uint32_t r = G * R; r < 32; ++r) {
                    for (uint32_t f = 0; f < 2; ++f) {
                        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                            wr + d * TB + ((r >> 4) * 2 + f) * 512 + (r & 15) * 32);
                        for (uint32_t w = 0; w < 8; ++w) {
                            z[w] = 0;
                        }
                    }
                }
            }
        }
        cb_wait_front(cb_out, DT);
        const uint32_t o = get_read_ptr(cb_out);
        for (uint32_t d = 0; d < DT; ++d) {
            for (uint32_t i = 0; i < R; ++i) {
                const uint32_t r = j * R + i;
                for (uint32_t f = 0; f < 2; ++f) {
                    noc_async_write(o + d * TB + ((i >> 4) * 2 + f) * 512 + (i & 15) * 32,
                                    get_noc_addr(lx, ly, wr + d * TB + ((r >> 4) * 2 + f) * 512 + (r & 15) * 32), 32);
                }
            }
        }
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(lx, ly, sem), 1);
        cb_pop_front(cb_out, DT);
        if (j == 0) {
            noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem), G);
            for (uint32_t d = 0; d < DT; ++d) {
                noc_async_write(wr + d * TB, ad.get_noc_addr(idx * DT + d), TB);
            }
            noc_async_write_barrier();
        }
        return;
    }
    const auto ap = TensorAccessor(p_args, p0_addr, 4);
    noc_async_read(ap.get_noc_addr(0), scratch, 4);
    noc_async_read_barrier();
    const uint32_t p0 = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    const uint32_t rows_t = S / 32;
    cb_wait_front(cb_out, DT);
    const uint32_t o = get_read_ptr(cb_out);
    uint32_t cur = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < R; ++i) {
        const uint32_t s = (p0 + i) % S;
        const uint32_t tr = s >> 5;
        if (tr != cur) {
            if (cur != 0xFFFFFFFFu) {
                for (uint32_t d = 0; d < DT; ++d) {
                    noc_async_write(wr + d * TB, ad.get_noc_addr((idx * rows_t + cur) * DT + d), TB);
                }
                noc_async_write_barrier();
            }
            for (uint32_t d = 0; d < DT; ++d) {
                noc_async_read(ad.get_noc_addr((idx * rows_t + tr) * DT + d), wr + d * TB, TB);
            }
            noc_async_read_barrier();
            cur = tr;
        }
        for (uint32_t d = 0; d < DT; ++d) {
            copy_row(o + d * TB, i, wr + d * TB, s & 31);
        }
    }
    for (uint32_t d = 0; d < DT; ++d) {
        noc_async_write(wr + d * TB, ad.get_noc_addr((idx * rows_t + cur) * DT + d), TB);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, DT);
#else
    compute_kernel_hw_startup(cb_x, cb_x, cb_out);
    if (kind == 2) {
        cb_wait_front(cb_x, DT);
        reconfig_data_format_srca(cb_x);
        pack_reconfig_data_format(cb_out);
        copy_tile_to_dst_init_short(cb_x);
        tile_regs_acquire();
        for (uint32_t d = 0; d < DT; ++d) {
            copy_tile(cb_x, d, d);
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, DT);
        tile_regs_wait();
        for (uint32_t d = 0; d < DT; ++d) {
            pack_tile(d, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, DT);
        cb_pop_front(cb_x, DT);
        return;
    }
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_eps, 1);
    cb_wait_front(cb_gm, DT);
    cb_wait_front(cb_c, DT);
    cb_wait_front(cb_s, DT);
    for (uint32_t h = 0; h < nh; ++h) {
        cb_wait_front(cb_x, DT);
        // ---- squares
        reconfig_data_format(cb_x, cb_x);
        pack_reconfig_data_format(cb_sq);
        mul_init(cb_x, cb_x);
        tile_regs_acquire();
        for (uint32_t d = 0; d < DT; ++d) {
            mul_tiles(cb_x, cb_x, d, d, d);
        }
        tile_regs_commit();
        cb_reserve_back(cb_sq, DT);
        tile_regs_wait();
        for (uint32_t d = 0; d < DT; ++d) {
            pack_tile(d, cb_sq);
        }
        tile_regs_release();
        cb_push_back(cb_sq, DT);
        // ---- mean of squares
        cb_wait_front(cb_sq, DT);
        reconfig_data_format(cb_scaler, cb_sq);
        tile_regs_acquire();
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_scaler, cb_ex2);
        for (uint32_t d = 0; d < DT; ++d) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_scaler, d, 0, 0);
        }
        reduce_uninit();
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv_w);
        tile_regs_commit();
        cb_pop_front(cb_sq, DT);
        cb_reserve_back(cb_ex2, 1);
        pack_reconfig_data_format(cb_ex2);
        tile_regs_wait();
        pack_tile(0, cb_ex2);
        tile_regs_release();
        cb_push_back(cb_ex2, 1);
        // ---- 1 / sqrt(mean + eps)
        cb_wait_front(cb_ex2, 1);
        reconfig_data_format(cb_ex2, cb_eps);
        tile_regs_acquire();
        add_init(cb_ex2, cb_eps);
        add_tiles(cb_ex2, cb_eps, 0, 0, 0);
        rsqrt_tile_init<false>();
        rsqrt_tile<false>(0);
        tile_regs_commit();
        cb_pop_front(cb_ex2, 1);
        cb_reserve_back(cb_r, 1);
        pack_reconfig_data_format(cb_r);
        tile_regs_wait();
        pack_tile(0, cb_r);
        tile_regs_release();
        cb_push_back(cb_r, 1);
        // ---- x r -> fp32, x gamma -> bf16
        cb_wait_front(cb_r, 1);
        reconfig_data_format(cb_x, cb_r);
        tile_regs_acquire();
        mul_bcast_cols_init(cb_x, cb_r);
        for (uint32_t d = 0; d < DT; ++d) {
            mul_tiles_bcast_cols(cb_x, cb_r, d, 0, d);
        }
        tile_regs_commit();
        cb_pop_front(cb_r, 1);
        cb_pop_front(cb_x, DT);
        cb_reserve_back(cb_fus, DT);
        pack_reconfig_data_format(cb_fus);
        tile_regs_wait();
        for (uint32_t d = 0; d < DT; ++d) {
            pack_tile(d, cb_fus);
        }
        tile_regs_release();
        cb_push_back(cb_fus, DT);
        cb_wait_front(cb_fus, DT);
        reconfig_data_format(cb_fus, cb_gm);
        tile_regs_acquire();
        mul_bcast_rows_init(cb_fus, cb_gm);
        for (uint32_t d = 0; d < DT; ++d) {
            mul_tiles_bcast_rows(cb_fus, cb_gm, d, d, d);
        }
        tile_regs_commit();
        cb_pop_front(cb_fus, DT);
        cb_reserve_back(cb_xn, DT);
        pack_reconfig_data_format(cb_xn);
        tile_regs_wait();
        for (uint32_t d = 0; d < DT; ++d) {
            pack_tile(d, cb_xn);
        }
        tile_regs_release();
        cb_push_back(cb_xn, DT);
        // ---- RoPE: bf16(xn C) + bf16(xn[(d + 2) % 4] S') -> bf16
        cb_wait_front(cb_xn, DT);
        reconfig_data_format(cb_xn, cb_c);
        tile_regs_acquire();
        mul_init(cb_xn, cb_c);
        for (uint32_t d = 0; d < DT; ++d) {
            mul_tiles(cb_xn, cb_c, d, d, d);
        }
        tile_regs_commit();
        cb_reserve_back(cb_cm, DT);
        pack_reconfig_data_format(cb_cm);
        tile_regs_wait();
        for (uint32_t d = 0; d < DT; ++d) {
            pack_tile(d, cb_cm);
        }
        tile_regs_release();
        cb_push_back(cb_cm, DT);
        reconfig_data_format(cb_xn, cb_s);
        tile_regs_acquire();
        mul_init(cb_xn, cb_s);
        for (uint32_t d = 0; d < DT; ++d) {
            mul_tiles(cb_xn, cb_s, (d + 2) % DT, d, d);
        }
        tile_regs_commit();
        cb_pop_front(cb_xn, DT);
        cb_reserve_back(cb_sm, DT);
        pack_reconfig_data_format(cb_sm);
        tile_regs_wait();
        for (uint32_t d = 0; d < DT; ++d) {
            pack_tile(d, cb_sm);
        }
        tile_regs_release();
        cb_push_back(cb_sm, DT);
        cb_wait_front(cb_cm, DT);
        cb_wait_front(cb_sm, DT);
        reconfig_data_format(cb_cm, cb_sm);
        tile_regs_acquire();
        add_init(cb_cm, cb_sm);
        for (uint32_t d = 0; d < DT; ++d) {
            add_tiles(cb_cm, cb_sm, d, d, d);
        }
        tile_regs_commit();
        cb_pop_front(cb_cm, DT);
        cb_pop_front(cb_sm, DT);
        cb_reserve_back(cb_out, DT);
        pack_reconfig_data_format(cb_out);
        tile_regs_wait();
        for (uint32_t d = 0; d < DT; ++d) {
            pack_tile(d, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, DT);
    }
#endif
}
