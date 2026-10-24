// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// GDN decode head step, compute (one unit = one user's value head). The recurrence is the KDA decode kernel's
// (ttnn/cpp/ttnn/operations/experimental/kda/recurrent_decode, per-head decay), with the gating in front and the
// gated RMSNorm behind it:
//   d = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b)            (SFPU, face 0 of four input tiles)
//   qn = q * scale / sqrt(q.q + eps), kn = k / sqrt(k.k + eps)
//   vread = d (kn . S), qsd = d (qn . S), u = beta (v - vread)                   (decay is a scalar: applied after)
//   S_new = d S + kn^T (x) u, o = qsd + (qn . kn) u
//   y = o / sqrt(mean(o^2) + eps) * w * silu(z)
// The state is bf16 in DRAM: exact as a matmul operand (no split) and in FP32 DST, packed back to bf16 once per step.
// The FP32 vector operands are split hi + lo as in the KDA kernel (see its header for the TF32 reasoning). Vectors live
// in row 0 of their tiles (zero elsewhere), so their SFPU work covers the first rows only (VectorMode::R, two iterations);
// every DST tile such an op writes is first loaded with a row-vector input tile, so the rows it leaves alone stay zero
// (the rank-1 update k^T (x) u sums over all rows of its operands);
// per-unit scalars (d, beta, the norms, qn . kn) are packed once and read back by value (read_tile_value) instead of
// being broadcast from a column tile.

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"

#ifdef TRISC_MATH
namespace ckernel::sfpu {
// DST[hi] = DST[x] with the low mantissa bits cleared (MASK), DST[lo] = DST[x] - DST[hi] (exact). hi may alias x.
template <std::uint32_t MASK, int ITERATIONS = 8>
inline void calculate_gdn_split(const std::uint32_t x, const std::uint32_t hi, const std::uint32_t lo) {
    constexpr std::uint32_t tile = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[x * tile];
        sfpi::vFloat h = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(v) & MASK);
        sfpi::dst_reg[hi * tile] = h;
        sfpi::dst_reg[lo * tile] = v - h;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

namespace {

constexpr uint32_t IN0_MASK = 0xFFFFE000u;  // hi = 11 significant bits (srcB, matmul in0)
constexpr uint32_t IN1_MASK = 0xFFFFC000u;  // hi = 10 significant bits (srcA, matmul in1)

// ---- SFPU on the row-vector tiles: faces 0 and 1, the first two of each face's eight SFPU iterations (they span row 0)
constexpr int ROW_ITERS = 2;
ALWI void split_init() { MATH((SFPU_BINARY_INIT(unused))); }
template <uint32_t MASK>
ALWI void row_split(uint32_t x, uint32_t hi, uint32_t lo) {
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_gdn_split, (MASK, ROW_ITERS), x, hi, lo, VectorMode::R)));
}
ALWI void row_scale(uint32_t idst, uint32_t bits) {  // after binop_with_scalar_tile_init
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_binop_with_scalar, (APPROX, MUL_UNARY, ROW_ITERS, DST_ACCUM_MODE), idst,
        VectorMode::R, bits)));
}
ALWI void row_add_scalar(uint32_t idst, uint32_t bits) {
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_binop_with_scalar, (APPROX, ADD_UNARY, ROW_ITERS, DST_ACCUM_MODE), idst,
        VectorMode::R, bits)));
}
ALWI void row_add(uint32_t a, uint32_t b, uint32_t o) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::ADD, ROW_ITERS, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default), a, b, o,
        VectorMode::R)));
}
ALWI void row_sub(uint32_t a, uint32_t b, uint32_t o) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::SUB, ROW_ITERS, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default), a, b, o,
        VectorMode::R)));
}
ALWI void row_mul(uint32_t a, uint32_t b, uint32_t o) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sfpu_binary_mul, (APPROX, ckernel::BinaryOp::MUL, ROW_ITERS, DST_ACCUM_MODE),
        a, b, o, VectorMode::R)));
}
ALWI void row_rsqrt(uint32_t idst) {
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_rsqrt, (APPROX, ROW_ITERS, DST_ACCUM_MODE, false, false), idst,
        VectorMode::R)));
}
ALWI void row_silu(uint32_t idst) {
    MATH((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_silu, (DST_ACCUM_MODE, ROW_ITERS), idst, VectorMode::R)));
}

// Inputs from the reader
constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_gin = 3, cb_S = 5, cb_z = 29, cb_w = 30;
// Outputs to the writer
constexpr uint32_t cb_state = 7, cb_y = 31;
// FP32 intermediates (UnpackToDestFp32)
constexpr uint32_t cb_vread = 10, cb_dcol = 13, cb_nrm = 16, cb_qsd = 25, cb_qk = 26;
// matmul operands (Default unpack)
constexpr uint32_t cb_o = 6, cb_uh = 11, cb_kcol = 12, cb_qnh = 14, cb_knh = 15;
constexpr uint32_t cb_xh = 18, cb_xl = 19, cb_qnl = 20, cb_knl = 21, cb_kcoll = 22, cb_ul = 23;
constexpr uint32_t cb_kn1h = 27, cb_kn1l = 28;

constexpr uint32_t DST_TILES = 4;  // FP32 DST tiles per acquire (half sync)

// FPU work (copy / matmul) after SFPU work on the same DST tiles in one acquire: the SFPU calls wait for the FPU before
// they start, nothing makes the FPU wait for the SFPU
ALWI void sfpu_drain() {
#ifdef TRISC_MATH
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
#endif
}

ALWI void copy_from(uint32_t cb) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
}

ALWI void mm3_acc(uint32_t ah, uint32_t al, uint32_t bh, uint32_t bl, uint32_t a_tile, uint32_t b_tile, uint32_t idst) {
    matmul_tiles(ah, bh, a_tile, b_tile, idst);
    matmul_tiles(ah, bl, a_tile, b_tile, idst);
    matmul_tiles(al, bh, a_tile, b_tile, idst);
}

// matmul in0 -> srcB, in1 -> srcA
ALWI void mm_begin(uint32_t in0, uint32_t in1, uint32_t transpose) {
    reconfig_data_format(in1, in0);
    matmul_init(in0, in1, transpose);
}

// n scalars from element 0 of the front tiles of cb (FP32), popped
template <uint32_t N>
ALWI void read_scalars(uint32_t cb, uint32_t (&out)[N]) {
    CircularBuffer c(cb);
    c.wait_front(N);
    UNPACK((invalidate_l1_cache()));  // the slots were read for an earlier unit: drop cached lines (Blackhole)
    for (uint32_t i = 0; i < N; i++) {
        out[i] = read_tile_value(cb, i, 0);
    }
    c.pop_front(N);
}

// d (tile 0) and beta (tile 1) of this unit -> cb_dcol; SFPU on face 0 only (the values sit in column 0)
void gates(uint32_t sp_beta, uint32_t sp_rbeta, uint32_t sp_thr) {
    CircularBuffer cg(cb_gin), cd(cb_dcol);
    cg.wait_front(4);
    cd.reserve_back(2);
    pack_reconfig_data_format(cb_dcol);
    tile_regs_acquire();
    copy_from(cb_gin);
    copy_tile(cb_gin, 0, 0);  // a
    copy_tile(cb_gin, 1, 1);  // dt_bias
    copy_tile(cb_gin, 2, 3);  // -exp(A_log)
    copy_tile(cb_gin, 3, 2);  // b
    add_binary_tile_init();
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::ADD, 8, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default), 0, 1, 0,
        VectorMode::None)));
    softplus_tile_init();
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_softplus, (APPROX, DST_ACCUM_MODE), 0, VectorMode::None, sp_beta,
        sp_rbeta, sp_thr)));
    mul_binary_tile_init();
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sfpu_binary_mul, (APPROX, ckernel::BinaryOp::MUL, 8, DST_ACCUM_MODE),
        0, 3, 0, VectorMode::None)));
    exp_tile_init();
    exp_tile(0, VectorMode::None);
    sigmoid_tile_init();
    sigmoid_tile<VectorMode::None>(2);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_dcol);
    pack_tile(2, cb_dcol);
    tile_regs_release();
    cd.push_back(2);
    cg.pop_front(4);
}

// x [Kt] row tiles -> xh / xl (in1 split) at tile offset `at`
template <uint32_t Kt>
void split_rows(uint32_t x, uint32_t at) {
    copy_from(x);
    split_init();
    for (uint32_t i = 0; i < Kt; i++) {
        tile_regs_acquire();
        copy_tile(x, i, 0);
        copy_tile(x, i, 1);
        row_split<IN1_MASK>(0, 0, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_xh, at + i);
        pack_tile<true>(1, cb_xl, at + i);
        tile_regs_release();
    }
}

// rq = scale / sqrt(q.q + eps) (hi part only: the output RMSNorm cancels q's scale), rk = 1 / sqrt(k.k + eps)
template <uint32_t Kt>
void norms(uint32_t eps_bits, uint32_t scale_bits, uint32_t& rq, uint32_t& rk) {
    CircularBuffer ch(cb_xh), cl(cb_xl), cn(cb_nrm);
    ch.reserve_back(2 * Kt);
    cl.reserve_back(2 * Kt);
    split_rows<Kt>(cb_q, 0);
    split_rows<Kt>(cb_k, Kt);
    ch.push_back(2 * Kt);
    cl.push_back(2 * Kt);
    ch.wait_front(2 * Kt);
    cl.wait_front(2 * Kt);
    cn.reserve_back(2);
    tile_regs_acquire();
    mm_begin(cb_xh, cb_xh, 1);
    for (uint32_t i = 0; i < Kt; i++) {
        matmul_tiles(cb_xh, cb_xh, i, i, 0);
    }
    for (uint32_t i = Kt; i < 2 * Kt; i++) {
        matmul_tiles(cb_xh, cb_xh, i, i, 1);
        matmul_tiles(cb_xl, cb_xh, i, i, 1);
        matmul_tiles(cb_xl, cb_xh, i, i, 1);
        matmul_tiles(cb_xl, cb_xl, i, i, 1);
    }
    binop_with_scalar_tile_init();
    row_add_scalar(0, eps_bits);
    row_add_scalar(1, eps_bits);
    rsqrt_tile_init<false>();
    row_rsqrt(0);
    row_rsqrt(1);
    binop_with_scalar_tile_init();
    row_scale(0, scale_bits);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_nrm);
    pack_tile(1, cb_nrm);
    tile_regs_release();
    cn.push_back(2);
    ch.pop_front(2 * Kt);
    cl.pop_front(2 * Kt);
    uint32_t r[2];
    read_scalars<2>(cb_nrm, r);
    rq = r[0];
    rk = r[1];
}

// qn = q rq -> qnh / qnl (in0); kn = k rk -> knh / knl (in0) and kn1h / kn1l (in1)
template <uint32_t Kt>
void scale_split(uint32_t rq, uint32_t rk) {
    CircularBuffer qh(cb_qnh), ql(cb_qnl), kh(cb_knh), kl(cb_knl), k1h(cb_kn1h), k1l(cb_kn1l);
    qh.reserve_back(Kt);
    ql.reserve_back(Kt);
    kh.reserve_back(Kt);
    kl.reserve_back(Kt);
    k1h.reserve_back(Kt);
    k1l.reserve_back(Kt);
    copy_from(cb_q);
    for (uint32_t i = 0; i < Kt; i++) {
        tile_regs_acquire();
        copy_tile(cb_q, i, 0);
        copy_tile(cb_q, i, 1);
        binop_with_scalar_tile_init();
        row_scale(0, rq);
        split_init();
        row_split<IN0_MASK>(0, 0, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_qnh, i);
        pack_tile<true>(1, cb_qnl, i);
        tile_regs_release();
    }
    for (uint32_t i = 0; i < Kt; i++) {
        tile_regs_acquire();
        copy_tile(cb_k, i, 0);
        copy_tile(cb_k, i, 1);
        copy_tile(cb_k, i, 2);
        copy_tile(cb_k, i, 3);
        binop_with_scalar_tile_init();
        row_scale(0, rk);
        split_init();
        row_split<IN0_MASK>(0, 1, 2);
        row_split<IN1_MASK>(0, 0, 3);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(1, cb_knh, i);
        pack_tile<true>(2, cb_knl, i);
        pack_tile<true>(0, cb_kn1h, i);
        pack_tile<true>(3, cb_kn1l, i);
        tile_regs_release();
    }
    qh.push_back(Kt);
    ql.push_back(Kt);
    kh.push_back(Kt);
    kl.push_back(Kt);
    k1h.push_back(Kt);
    k1l.push_back(Kt);
}

// kn . S -> cb_vread, qn . S -> cb_qsd (undecayed; S is bf16, exact in srcA)
template <uint32_t Kt, uint32_t Vt>
void read_state() {
    CircularBuffer cv(cb_vread), cq(cb_qsd);
    cv.reserve_back(Vt);
    cq.reserve_back(Vt);
    mm_begin(cb_knh, cb_S, 0);
    for (uint32_t ni = 0; ni < Vt; ni++) {
        tile_regs_acquire();
        for (uint32_t ki = 0; ki < Kt; ki++) {
            const uint32_t s = ki * Vt + ni;
            matmul_tiles(cb_knh, cb_S, ki, s, 0);
            matmul_tiles(cb_knl, cb_S, ki, s, 0);
            matmul_tiles(cb_qnh, cb_S, ki, s, 1);
            matmul_tiles(cb_qnl, cb_S, ki, s, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_vread, ni);
        pack_tile<true>(1, cb_qsd, ni);
        tile_regs_release();
    }
    cv.push_back(Vt);
    cq.push_back(Vt);
}

template <uint32_t Kt>
uint32_t q_dot_k() {
    CircularBuffer c(cb_qk);
    c.reserve_back(1);
    tile_regs_acquire();
    mm_begin(cb_qnh, cb_kn1h, 1);
    for (uint32_t ki = 0; ki < Kt; ki++) {
        mm3_acc(cb_qnh, cb_qnl, cb_kn1h, cb_kn1l, ki, ki, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_qk);
    tile_regs_release();
    c.push_back(1);
    uint32_t r[1];
    read_scalars<1>(cb_qk, r);
    return r[0];
}

// u = beta (v - d vread) (split to uh / ul) and o = d qsd + qk u -> cb_o
template <uint32_t Vt>
void delta_and_output(uint32_t d, uint32_t beta, uint32_t qk) {
    CircularBuffer ch(cb_uh), cl(cb_ul), co(cb_o);
    ch.reserve_back(Vt);
    cl.reserve_back(Vt);
    co.reserve_back(Vt);
    copy_from(cb_v);
    for (uint32_t j = 0; j < Vt; j++) {
        tile_regs_acquire();
        copy_tile(cb_v, j, 0);
        copy_tile(cb_vread, j, 1);
        copy_tile(cb_v, j, 2);
        copy_tile(cb_qsd, j, 3);
        binop_with_scalar_tile_init();
        row_scale(1, d);
        row_scale(3, d);
        sub_binary_tile_init();
        row_sub(0, 1, 0);
        binop_with_scalar_tile_init();
        row_scale(0, beta);
        split_init();
        row_split<IN1_MASK>(0, 1, 2);
        binop_with_scalar_tile_init();
        row_scale(0, qk);
        add_binary_tile_init();
        row_add(0, 3, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_o, j);
        pack_tile<true>(1, cb_uh, j);
        pack_tile<true>(2, cb_ul, j);
        tile_regs_release();
    }
    ch.push_back(Vt);
    cl.push_back(Vt);
    co.push_back(Vt);
}

void transpose_block(uint32_t in, uint32_t o, uint32_t n) {
    CircularBuffer cb_o_(o);
    cb_o_.reserve_back(n);
    reconfig_data_format_srca(in);
    transpose_init(in);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        transpose_tile(in, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_o_.push_back(n);
}

// S_new = d S + kcol (x) u -> cb_state (bf16)
template <uint32_t Kt, uint32_t Vt>
void rank1_update(uint32_t d) {
    constexpr uint32_t kv = Kt * Vt;
    CircularBuffer cb_st(cb_state);
    cb_st.reserve_back(kv);
    pack_reconfig_data_format(cb_state);
    for (uint32_t mi = 0; mi < Kt; mi++) {
        for (uint32_t n0 = 0; n0 < Vt; n0 += DST_TILES) {
            const uint32_t nn = (Vt - n0 < DST_TILES) ? (Vt - n0) : DST_TILES;
            tile_regs_acquire();
            copy_from(cb_S);
            for (uint32_t j = 0; j < nn; j++) {
                copy_tile(cb_S, mi * Vt + n0 + j, j);
            }
            binop_with_scalar_tile_init();
            for (uint32_t j = 0; j < nn; j++) {
                mul_unary_tile(j, d);
            }
            sfpu_drain();
            mm_begin(cb_kcol, cb_uh, 0);
            for (uint32_t j = 0; j < nn; j++) {
                mm3_acc(cb_kcol, cb_kcoll, cb_uh, cb_ul, mi, n0 + j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < nn; j++) {
                pack_tile<true>(j, cb_state, mi * Vt + n0 + j);
            }
            tile_regs_release();
        }
    }
    cb_st.push_back(kv);
    pack_reconfig_data_format(cb_xh);
}

// y = o / sqrt(mean(o^2) + eps) * w * silu(z) -> cb_y (bf16)
template <uint32_t Vt>
void norm_gate(uint32_t eps_bits, uint32_t inv_dv_bits) {
    // the scalar goes through cb_qk (1 tile): cb_nrm takes 2-tile pushes only (a push must not wrap the CB)
    CircularBuffer co(cb_o), cz(cb_z), cn(cb_qk), cy(cb_y);
    co.wait_front(Vt);
    cz.wait_front(Vt);
    cn.reserve_back(1);
    tile_regs_acquire();
    mm_begin(cb_o, cb_o, 1);
    for (uint32_t j = 0; j < Vt; j++) {
        matmul_tiles(cb_o, cb_o, j, j, 0);
    }
    binop_with_scalar_tile_init();
    row_scale(0, inv_dv_bits);
    row_add_scalar(0, eps_bits);
    rsqrt_tile_init<false>();
    row_rsqrt(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_qk);
    tile_regs_release();
    cn.push_back(1);
    uint32_t r[1];
    read_scalars<1>(cb_qk, r);
    cy.reserve_back(Vt);
    pack_reconfig_data_format(cb_y);
    for (uint32_t j = 0; j < Vt; j++) {
        tile_regs_acquire();
        copy_from(cb_o);
        copy_tile(cb_o, j, 0);
        binop_with_scalar_tile_init();
        row_scale(0, r[0]);
        sfpu_drain();
        copy_from(cb_w);
        copy_tile(cb_w, j, 1);
        mul_binary_tile_init();
        row_mul(0, 1, 0);
        sfpu_drain();
        copy_from(cb_z);
        copy_tile(cb_z, j, 1);
        silu_tile_init();
        row_silu(1);
        mul_binary_tile_init();
        row_mul(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_y);
        tile_regs_release();
    }
    cy.push_back(Vt);
    pack_reconfig_data_format(cb_xh);
    co.pop_front(Vt);
    cz.pop_front(Vt);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(2);      // q / k L2 norm eps
    constexpr uint32_t q_scale_bits = get_compile_time_arg_val(3);
    constexpr uint32_t norm_eps_bits = get_compile_time_arg_val(4);  // gated RMSNorm eps
    constexpr uint32_t inv_dv_bits = get_compile_time_arg_val(5);    // 1 / Dv
    constexpr uint32_t sp_beta = get_compile_time_arg_val(6);        // softplus beta, 1 / beta, threshold
    constexpr uint32_t sp_rbeta = get_compile_time_arg_val(7);
    constexpr uint32_t sp_thr = get_compile_time_arg_val(8);
    constexpr uint32_t kv = Kt * Vt;
    const uint32_t count = get_arg_val<uint32_t>(0);
    if (count == 0) {
        return;
    }

    compute_kernel_hw_startup(cb_xh, cb_xh, cb_xh);
    CircularBuffer q(cb_q), k(cb_k), v(cb_v), S(cb_S), w(cb_w);
    CircularBuffer qnh(cb_qnh), qnl(cb_qnl), knh(cb_knh), knl(cb_knl), vread(cb_vread), qsd(cb_qsd);
    CircularBuffer uh(cb_uh), ul(cb_ul), kcol(cb_kcol), kcoll(cb_kcoll), kn1h(cb_kn1h), kn1l(cb_kn1l);
    w.wait_front(Vt);

    for (uint32_t it = 0; it < count; it++) {
        uint32_t g[2];
        {
            DeviceZoneScopedN("gates");
            gates(sp_beta, sp_rbeta, sp_thr);
            read_scalars<2>(cb_dcol, g);
        }
        const uint32_t d = g[0], beta = g[1];

        {
            DeviceZoneScopedN("l2norm");
            q.wait_front(Kt);
            k.wait_front(Kt);
            uint32_t rq, rk;
            norms<Kt>(eps_bits, q_scale_bits, rq, rk);
            scale_split<Kt>(rq, rk);
            q.pop_front(Kt);
            k.pop_front(Kt);
        }
        qnh.wait_front(Kt);
        qnl.wait_front(Kt);
        knh.wait_front(Kt);
        knl.wait_front(Kt);

        S.wait_front(kv);
        {
            DeviceZoneScopedN("read_state");
            read_state<Kt, Vt>();
        }
        kn1h.wait_front(Kt);
        kn1l.wait_front(Kt);
        uint32_t qk;
        {
            DeviceZoneScopedN("q_dot_k");
            qk = q_dot_k<Kt>();
        }
        qnh.pop_front(Kt);
        qnl.pop_front(Kt);
        kn1h.pop_front(Kt);
        kn1l.pop_front(Kt);

        vread.wait_front(Vt);
        qsd.wait_front(Vt);
        v.wait_front(Vt);
        {
            DeviceZoneScopedN("delta");
            delta_and_output<Vt>(d, beta, qk);
        }
        v.pop_front(Vt);
        vread.pop_front(Vt);
        qsd.pop_front(Vt);
        uh.wait_front(Vt);
        ul.wait_front(Vt);

        {
            DeviceZoneScopedN("transpose");
            transpose_block(cb_knh, cb_kcol, Kt);
            transpose_block(cb_knl, cb_kcoll, Kt);
        }
        knh.pop_front(Kt);
        knl.pop_front(Kt);
        kcol.wait_front(Kt);
        kcoll.wait_front(Kt);
        {
            DeviceZoneScopedN("rank1");
            rank1_update<Kt, Vt>(d);
        }
        kcol.pop_front(Kt);
        kcoll.pop_front(Kt);
        uh.pop_front(Vt);
        ul.pop_front(Vt);
        S.pop_front(kv);

        {
            DeviceZoneScopedN("norm_gate");
            norm_gate<Vt>(norm_eps_bits, inv_dv_bits);
        }
    }
    w.pop_front(Vt);
}
