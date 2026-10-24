// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// GDN decode head step, compute (one unit = one user's value head). The recurrence is the KDA decode kernel's
// (ttnn/cpp/ttnn/operations/experimental/kda/recurrent_decode, per-head decay), with the gating in front and the
// gated RMSNorm behind it:
//   decay = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b)       (SFPU, column 0 of four input tiles)
//   qn = q * scale / sqrt(q.q + eps), kn = k / sqrt(k.k + eps)
//   sd = decay S, vread = kn . sd, qsd = qn . sd, u = beta (v - vread)
//   S_new = sd + kn^T (x) u, o = qsd + (qn . kn) u
//   y = o / sqrt(mean(o^2) + eps) * w * silu(z)
// The state is bf16 in DRAM (unpacked exactly into FP32 DST, packed back to bf16 once per step); every FP32 matmul
// operand is split hi + lo as in the KDA kernel (see its header for the TF32 reasoning).

#include <cstdint>

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
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/dataflow/circular_buffer.h"

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

ALWI void split_init() { MATH((SFPU_BINARY_INIT(unused))); }
template <uint32_t MASK>
ALWI void split_tile(uint32_t x, uint32_t hi, uint32_t lo) {
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_gdn_split, (MASK, 8), x, hi, lo, VectorMode::RC)));
}

// Inputs from the reader
constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_gin = 3, cb_S = 5, cb_z = 29, cb_w = 30;
// Outputs to the writer
constexpr uint32_t cb_state = 7, cb_y = 31;
// FP32 intermediates (UnpackToDestFp32)
constexpr uint32_t cb_beta = 4, cb_vread = 10, cb_dcol = 13, cb_nrm = 16, cb_sdx = 24, cb_qsd = 25, cb_qk = 26;
// matmul operands (Default unpack)
constexpr uint32_t cb_o = 6, cb_sdh = 8, cb_sdl = 9, cb_uh = 11, cb_kcol = 12, cb_qnh = 14, cb_knh = 15;
constexpr uint32_t cb_xh = 18, cb_xl = 19, cb_qnl = 20, cb_knl = 21, cb_kcoll = 22, cb_ul = 23;
constexpr uint32_t cb_kn1h = 27, cb_kn1l = 28;

constexpr uint32_t DST_TILES = 4;  // FP32 DST tiles per acquire (half sync)

ALWI void copy_from(uint32_t cb) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
}

ALWI void mm3_acc(uint32_t ah, uint32_t al, uint32_t bh, uint32_t bl, uint32_t a_tile, uint32_t b_tile, uint32_t idst) {
    matmul_tiles(ah, bh, a_tile, b_tile, idst);
    matmul_tiles(ah, bl, a_tile, b_tile, idst);
    matmul_tiles(al, bh, a_tile, b_tile, idst);
}

ALWI void mm_begin(uint32_t a, uint32_t b, uint32_t transpose) {
    reconfig_data_format(a, b);
    matmul_init(a, b, transpose);
}

// decay (column 0, every row) -> cb_dcol, beta (column 0) -> cb_beta
void gates(uint32_t sp_beta, uint32_t sp_rbeta, uint32_t sp_thr) {
    CircularBuffer cg(cb_gin), cd(cb_dcol), cbt(cb_beta);
    cg.wait_front(4);
    cd.reserve_back(1);
    cbt.reserve_back(1);
    pack_reconfig_data_format(cb_dcol);
    tile_regs_acquire();
    copy_from(cb_gin);
    copy_tile(cb_gin, 0, 0);  // a
    copy_tile(cb_gin, 1, 1);  // dt_bias
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    softplus_tile_init();
    softplus_tile(0, sp_beta, sp_rbeta, sp_thr);
    copy_tile(cb_gin, 2, 1);  // -exp(A_log)
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    exp_tile_init();
    exp_tile(0);
    copy_tile(cb_gin, 3, 2);  // b
    sigmoid_tile_init();
    sigmoid_tile(2);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_dcol);
    pack_tile(2, cb_beta);
    tile_regs_release();
    cd.push_back(1);
    cbt.push_back(1);
    cg.pop_front(4);
}

// x [Kt] row tiles -> xh / xl, split for the matmul in1 role
template <uint32_t Kt>
void split_rows(uint32_t x) {
    CircularBuffer ch(cb_xh), cl(cb_xl);
    ch.reserve_back(Kt);
    cl.reserve_back(Kt);
    for (uint32_t i = 0; i < Kt; i++) {
        tile_regs_acquire();
        copy_from(x);
        copy_tile(x, i, 0);
        split_init();
        split_tile<IN1_MASK>(0, 1, 2);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(1, cb_xh, i);
        pack_tile<true>(2, cb_xl, i);
        tile_regs_release();
    }
    ch.push_back(Kt);
    cl.push_back(Kt);
}

// xn = x * scale / sqrt(x . x + eps), split for in0 into nh / nl and (IN1) for in1 into cb_kn1h / cb_kn1l
template <uint32_t Kt, bool IN1>
void l2_normalize(uint32_t x, uint32_t nh, uint32_t nl, uint32_t eps_bits, uint32_t scale_bits, bool apply_scale) {
    split_rows<Kt>(x);
    CircularBuffer ch(cb_xh), cl(cb_xl), cn(cb_nrm), cnh(nh), cnl(nl);
    ch.wait_front(Kt);
    cl.wait_front(Kt);
    cn.reserve_back(1);
    tile_regs_acquire();
    mm_begin(cb_xh, cb_xh, 1);
    for (uint32_t i = 0; i < Kt; i++) {
        matmul_tiles(cb_xh, cb_xh, i, i, 0);
        matmul_tiles(cb_xl, cb_xh, i, i, 0);
        matmul_tiles(cb_xl, cb_xh, i, i, 0);
        matmul_tiles(cb_xl, cb_xl, i, i, 0);
    }
    binop_with_scalar_tile_init();
    add_unary_tile(0, eps_bits);
    rsqrt_tile_init<false>();
    rsqrt_tile<false, false>(0);
    if (apply_scale) {
        binop_with_scalar_tile_init();
        mul_unary_tile(0, scale_bits);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_nrm);
    tile_regs_release();
    cn.push_back(1);
    ch.pop_front(Kt);
    cl.pop_front(Kt);

    cn.wait_front(1);
    CircularBuffer c1h(cb_kn1h), c1l(cb_kn1l);
    cnh.reserve_back(Kt);
    cnl.reserve_back(Kt);
    if constexpr (IN1) {
        c1h.reserve_back(Kt);
        c1l.reserve_back(Kt);
    }
    for (uint32_t i = 0; i < Kt; i++) {
        tile_regs_acquire();
        copy_from(x);
        copy_tile(x, i, 0);
        copy_from(cb_nrm);
        copy_tile(cb_nrm, 0, 3);
        sfpu_mul_bcast_col_init();
        sfpu_mul_bcast_col(0, 3);
        split_init();
        split_tile<IN0_MASK>(0, 1, 2);
        if constexpr (IN1) {
            split_tile<IN1_MASK>(0, 0, 3);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(1, nh, i);
        pack_tile<true>(2, nl, i);
        if constexpr (IN1) {
            pack_tile<true>(0, cb_kn1h, i);
            pack_tile<true>(3, cb_kn1l, i);
        }
        tile_regs_release();
    }
    cnh.push_back(Kt);
    cnl.push_back(Kt);
    if constexpr (IN1) {
        c1h.push_back(Kt);
        c1l.push_back(Kt);
    }
    cn.pop_front(1);
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

// sd = decay * S (bf16 state, exact in DST; decay from column 0 of cb_dcol): packed exact (sdx) and split (sdh / sdl)
template <uint32_t Kt, uint32_t Vt>
void decay_state() {
    constexpr uint32_t kv = Kt * Vt;
    CircularBuffer cx(cb_sdx), ch(cb_sdh), cl(cb_sdl);
    cx.reserve_back(kv);
    ch.reserve_back(kv);
    cl.reserve_back(kv);
    for (uint32_t t = 0; t < kv; t++) {
        tile_regs_acquire();
        copy_from(cb_S);
        copy_tile(cb_S, t, 0);
        copy_from(cb_dcol);
        copy_tile(cb_dcol, 0, 3);
        sfpu_mul_bcast_col_init();
        sfpu_mul_bcast_col(0, 3);
        split_init();
        split_tile<IN1_MASK>(0, 1, 2);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_sdx, t);
        pack_tile<true>(1, cb_sdh, t);
        pack_tile<true>(2, cb_sdl, t);
        tile_regs_release();
    }
    cx.push_back(kv);
    ch.push_back(kv);
    cl.push_back(kv);
}

template <uint32_t Kt, uint32_t Vt>
void read_state() {
    CircularBuffer cv(cb_vread), cq(cb_qsd);
    cv.reserve_back(Vt);
    cq.reserve_back(Vt);
    mm_begin(cb_knh, cb_sdh, 0);
    for (uint32_t ni = 0; ni < Vt; ni++) {
        tile_regs_acquire();
        for (uint32_t ki = 0; ki < Kt; ki++) {
            mm3_acc(cb_knh, cb_knl, cb_sdh, cb_sdl, ki, ki * Vt + ni, 0);
            mm3_acc(cb_qnh, cb_qnl, cb_sdh, cb_sdl, ki, ki * Vt + ni, 1);
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
void q_dot_k() {
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
}

// u = beta * (v - vread) (split to uh / ul) and o = qsd + qk * u -> cb_o
template <uint32_t Vt>
void delta_and_output() {
    CircularBuffer ch(cb_uh), cl(cb_ul), co(cb_o);
    ch.reserve_back(Vt);
    cl.reserve_back(Vt);
    co.reserve_back(Vt);
    for (uint32_t j = 0; j < Vt; j++) {
        tile_regs_acquire();
        copy_from(cb_v);
        copy_tile(cb_v, j, 0);
        copy_tile(cb_vread, j, 1);
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);
        copy_tile(cb_beta, 0, 3);
        sfpu_mul_bcast_col_init();
        sfpu_mul_bcast_col(0, 3);
        split_init();
        split_tile<IN1_MASK>(0, 1, 2);
        copy_tile(cb_qk, 0, 3);
        sfpu_mul_bcast_col_init();
        sfpu_mul_bcast_col(0, 3);
        copy_tile(cb_qsd, j, 3);
        add_binary_tile_init();
        add_binary_tile(0, 3, 0);
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

// S_new = sd + kcol (x) u -> cb_state (bf16)
template <uint32_t Kt, uint32_t Vt>
void rank1_update() {
    constexpr uint32_t kv = Kt * Vt;
    CircularBuffer cb_st(cb_state);
    cb_st.reserve_back(kv);
    pack_reconfig_data_format(cb_state);
    for (uint32_t mi = 0; mi < Kt; mi++) {
        for (uint32_t n0 = 0; n0 < Vt; n0 += DST_TILES) {
            const uint32_t nn = (Vt - n0 < DST_TILES) ? (Vt - n0) : DST_TILES;
            tile_regs_acquire();
            copy_from(cb_sdx);
            for (uint32_t j = 0; j < nn; j++) {
                copy_tile(cb_sdx, mi * Vt + n0 + j, j);
            }
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
    CircularBuffer co(cb_o), cz(cb_z), cn(cb_nrm), cy(cb_y);
    co.wait_front(Vt);
    cz.wait_front(Vt);
    cn.reserve_back(1);
    tile_regs_acquire();
    mm_begin(cb_o, cb_o, 1);
    for (uint32_t j = 0; j < Vt; j++) {
        matmul_tiles(cb_o, cb_o, j, j, 0);
    }
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_dv_bits);
    add_unary_tile(0, eps_bits);
    rsqrt_tile_init<false>();
    rsqrt_tile<false, false>(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_nrm);
    tile_regs_release();
    cn.push_back(1);
    cn.wait_front(1);
    cy.reserve_back(Vt);
    pack_reconfig_data_format(cb_y);
    for (uint32_t j = 0; j < Vt; j++) {
        tile_regs_acquire();
        copy_from(cb_o);
        copy_tile(cb_o, j, 0);
        copy_from(cb_nrm);
        copy_tile(cb_nrm, 0, 1);
        sfpu_mul_bcast_col_init();
        sfpu_mul_bcast_col(0, 1);
        copy_from(cb_w);
        copy_tile(cb_w, j, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        copy_from(cb_z);
        copy_tile(cb_z, j, 1);
        silu_tile_init();
        silu_tile(1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_y);
        tile_regs_release();
    }
    cy.push_back(Vt);
    pack_reconfig_data_format(cb_xh);
    cn.pop_front(1);
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
    CircularBuffer qnh(cb_qnh), qnl(cb_qnl), knh(cb_knh), knl(cb_knl), dcol(cb_dcol), beta(cb_beta), sdx(cb_sdx);
    CircularBuffer sdh(cb_sdh), sdl(cb_sdl), vread(cb_vread), qsd(cb_qsd), qk(cb_qk), uh(cb_uh), ul(cb_ul);
    CircularBuffer kcol(cb_kcol), kcoll(cb_kcoll), kn1h(cb_kn1h), kn1l(cb_kn1l);
    w.wait_front(Vt);

    for (uint32_t it = 0; it < count; it++) {
        gates(sp_beta, sp_rbeta, sp_thr);

        q.wait_front(Kt);
        l2_normalize<Kt, false>(cb_q, cb_qnh, cb_qnl, eps_bits, q_scale_bits, true);
        q.pop_front(Kt);
        k.wait_front(Kt);
        l2_normalize<Kt, true>(cb_k, cb_knh, cb_knl, eps_bits, q_scale_bits, false);
        k.pop_front(Kt);
        qnh.wait_front(Kt);
        qnl.wait_front(Kt);
        knh.wait_front(Kt);
        knl.wait_front(Kt);

        dcol.wait_front(1);
        S.wait_front(kv);
        decay_state<Kt, Vt>();
        S.pop_front(kv);
        dcol.pop_front(1);
        sdh.wait_front(kv);
        sdl.wait_front(kv);

        read_state<Kt, Vt>();
        sdh.pop_front(kv);
        sdl.pop_front(kv);
        kn1h.wait_front(Kt);
        kn1l.wait_front(Kt);
        q_dot_k<Kt>();
        qnh.pop_front(Kt);
        qnl.pop_front(Kt);
        kn1h.pop_front(Kt);
        kn1l.pop_front(Kt);

        vread.wait_front(Vt);
        qsd.wait_front(Vt);
        qk.wait_front(1);
        v.wait_front(Vt);
        beta.wait_front(1);
        delta_and_output<Vt>();
        v.pop_front(Vt);
        vread.pop_front(Vt);
        qsd.pop_front(Vt);
        qk.pop_front(1);
        beta.pop_front(1);
        uh.wait_front(Vt);
        ul.wait_front(Vt);

        transpose_block(cb_knh, cb_kcol, Kt);
        transpose_block(cb_knl, cb_kcoll, Kt);
        knh.pop_front(Kt);
        knl.pop_front(Kt);
        kcol.wait_front(Kt);
        kcoll.wait_front(Kt);
        sdx.wait_front(kv);
        rank1_update<Kt, Vt>();
        kcol.pop_front(Kt);
        kcoll.pop_front(Kt);
        uh.pop_front(Vt);
        ul.pop_front(Vt);
        sdx.pop_front(kv);

        norm_gate<Vt>(norm_eps_bits, inv_dv_bits);
    }
    w.pop_front(Vt);
}
