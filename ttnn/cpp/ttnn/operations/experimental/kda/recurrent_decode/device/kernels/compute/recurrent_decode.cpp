// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// One KDA decode step per head (derived from the compute kernel of ttnn.transformer.fused_recurrent_gated_delta_rule;
// here the decay is per key channel and the q / k L2 norms run in-kernel):
//   qn    = q * q_scale / sqrt(q.q + eps),  kn = k / sqrt(k.k + eps)
//   sd    = diag(decay) S                 (decay row transposed to a column, column-broadcast multiply)
//   vread = kn . sd,  qsd = qn . sd       ([1,V])
//   u     = beta * (v - vread)            ([1,V])
//   S_new = sd + kn^T (x) u               (rank-1 update accumulated onto sd in DST)
//   o     = qn . S_new = qsd + (qn . kn) u
// Head rows are [1, D] tiles (row 0 valid, rows 1..31 zero), so x . x^T lands the squared norm in element [0,0] and
// the inner-dim-1 outer product is a matmul whose 31 padding lanes are zero.
//
// fp32 precision: the FPU reads fp32 at TF32 (10-bit mantissa, truncated), and its multiplier uses all 11 significant
// bits of srcB (matmul in0) but only the top 10 of srcA (matmul in1). An fp32 operand fed to it directly loses ~2^-11
// relative, biased; on the recurrent state that compounds every step. Here
//   * every elementwise op on fp32 data (decay * S, v - vread, beta * ., the output combine) runs on the SFPU on DST
//     tiles unpacked straight to DST (the "X" CBs below are UnpackToDestFp32, see the program factory);
//   * every matmul operand is split x = hi + lo in DST (lo = x - hi, exact): in0 operands at the TF32 boundary (hi =
//     11 significant bits), in1 operands one bit higher (hi = 10 significant bits), so each hi is used whole by the
//     multiplier; a.b = a_hi.b_hi + a_hi.b_lo + a_lo.b_hi in fp32 DST is then good to ~2^-19 instead of ~2^-11;
//   * the state seed of the rank-1 update is the exact fp32 sd (UnpackToDestFp32 copy), so S_new = sd + k^T u keeps
//     full precision.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/dataflow/circular_buffer.h"

#ifdef TRISC_MATH
namespace ckernel::sfpu {
// DST[hi] = DST[x] with the low mantissa bits cleared (MASK), DST[lo] = DST[x] - DST[hi] (exact). hi may alias x.
template <std::uint32_t MASK, int ITERATIONS = 8>
inline void calculate_kda_split(const std::uint32_t x, const std::uint32_t hi, const std::uint32_t lo) {
    constexpr std::uint32_t tile = 32;  // sfpi rows per DST tile
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

constexpr uint32_t IN0_MASK = 0xFFFFE000u;  // hi = 11 significant bits: srcB (matmul in0) uses all of them
constexpr uint32_t IN1_MASK = 0xFFFFC000u;  // hi = 10 significant bits: srcA (matmul in1) uses all of them

ALWI void split_init() { MATH((SFPU_BINARY_INIT(unused))); }
// DST[hi] + DST[lo] == DST[x] exactly; MASK picks the operand role (IN0_MASK / IN1_MASK)
template <uint32_t MASK>
ALWI void split_tile(uint32_t x, uint32_t hi, uint32_t lo) {
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_kda_split, (MASK, 8), x, hi, lo, VectorMode::RC)));
}

// Inputs (UnpackToDestFp32 unless noted)
constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_decay = 3, cb_beta = 4, cb_S = 5;
// Outputs (to the writer)
constexpr uint32_t cb_out = 6, cb_state = 7;
// Matmul operands (Default unpack: read at TF32; they only ever hold hi / lo halves, both TF32-exact or ~exact)
constexpr uint32_t cb_sdh = 8, cb_sdl = 9, cb_uh = 11, cb_kcol = 12, cb_qnh = 14, cb_knh = 15;
constexpr uint32_t cb_xh = 18, cb_xl = 19, cb_qnl = 20, cb_knl = 21, cb_kcoll = 22, cb_ul = 23;
// fp32 intermediates (UnpackToDestFp32)
constexpr uint32_t cb_vread = 10, cb_dcol = 13, cb_nrm = 16, cb_sdx = 24, cb_qsd = 25, cb_qk = 26;
// kn split for its matmul-in1 role (qn . kn^T)
constexpr uint32_t cb_kn1h = 27, cb_kn1l = 28;

constexpr uint32_t DST_TILES = 4;  // fp32 DST tiles per tile_regs_acquire (half-sync)

ALWI void copy_from(uint32_t cb) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
}

// DST[0][0,0] += sum over the n tiles of a . b (transpose: a . b^T), with both operands split: hi.hi + hi.lo + lo.hi
ALWI void mm3_acc(uint32_t ah, uint32_t al, uint32_t bh, uint32_t bl, uint32_t a_tile, uint32_t b_tile, uint32_t idst) {
    matmul_tiles(ah, bh, a_tile, b_tile, idst);
    matmul_tiles(ah, bl, a_tile, b_tile, idst);
    matmul_tiles(al, bh, a_tile, b_tile, idst);
}

ALWI void mm_begin(uint32_t a, uint32_t b, uint32_t transpose) {
    reconfig_data_format(a, b);
    matmul_init(a, b, transpose);
}

// x [Kt] row tiles (exact CB) -> xh / xl (pushed, Kt tiles each), split for both matmul roles (IN1_MASK).
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

// xn = x * scale / sqrt(x . x + eps), split for the in0 role into nh / nl (Kt tiles each) and, with IN1, also for
// the in1 role into cb_kn1h / cb_kn1l. x (exact CB) must be available; not popped.
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
        // x.x = hi.hi + 2 lo.hi + lo.lo: lo only ever reaches the multiplier through srcB (in0) except in lo.lo
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
        sfpu_mul_bcast_col(0, 3);  // row 0 * nrm[0,0]
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

// out[Kt] = per-tile transpose of in[Kt] (a [1,K] row becomes a [K,1] column). in must be available.
void transpose_block(uint32_t in, uint32_t o, uint32_t n) {
    CircularBuffer cb_o(o);
    cb_o.reserve_back(n);
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
    cb_o.push_back(n);
}

// sd[i, j] = S[i, j] * dcol[i] (SFPU, exact): packed exact (sdx, the rank-1 seed) and split (sdh / sdl, matmul
// operands).
template <uint32_t Kt, uint32_t Vt>
void decay_state() {
    constexpr uint32_t kv = Kt * Vt;
    CircularBuffer cx(cb_sdx), ch(cb_sdh), cl(cb_sdl);
    cx.reserve_back(kv);
    ch.reserve_back(kv);
    cl.reserve_back(kv);
    for (uint32_t i = 0; i < Kt; i++) {
        for (uint32_t j = 0; j < Vt; j++) {
            const uint32_t t = i * Vt + j;
            tile_regs_acquire();
            copy_from(cb_S);
            copy_tile(cb_S, t, 0);
            copy_tile(cb_dcol, i, 3);  // same (exact fp32) format as cb_S
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
    }
    cx.push_back(kv);
    ch.push_back(kv);
    cl.push_back(kv);
}

// vread = kn . sd -> cb_vread, qsd = qn . sd -> cb_qsd (both [1,Vt], exact fp32 out)
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

// qk[0,0] = qn . kn
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

// u = beta * (v - vread) (split to uh / ul) and o = qsd + qk * u -> cb_out, all on DST (SFPU).
template <uint32_t Vt>
void delta_and_output() {
    CircularBuffer ch(cb_uh), cl(cb_ul), co(cb_out);
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
        sfpu_mul_bcast_col(0, 3);  // u (row 0) = (v - vread) * beta[0,0]
        split_init();
        split_tile<IN1_MASK>(0, 1, 2);
        copy_tile(cb_qk, 0, 3);
        sfpu_mul_bcast_col_init();
        sfpu_mul_bcast_col(0, 3);  // qk * u
        copy_tile(cb_qsd, j, 3);
        add_binary_tile_init();
        add_binary_tile(0, 3, 0);  // o = qsd + qk * u
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_out, j);
        pack_tile<true>(1, cb_uh, j);
        pack_tile<true>(2, cb_ul, j);
        tile_regs_release();
    }
    ch.push_back(Vt);
    cl.push_back(Vt);
    co.push_back(Vt);
}

// S_new[Kt,Vt] = sd + kcol[Kt,1] (x) u[1,Vt]: each DST tile is seeded with the exact sd and the split rank-1 matmuls
// accumulate onto it; the result goes to cb_state (written to DRAM).
template <uint32_t Kt, uint32_t Vt>
void rank1_update() {
    constexpr uint32_t kv = Kt * Vt;
    CircularBuffer cb_st(cb_state);
    cb_st.reserve_back(kv);
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
}

}  // namespace

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(2);
    constexpr uint32_t q_scale_bits = get_compile_time_arg_val(3);
    constexpr uint32_t kv = Kt * Vt;
    const uint32_t count = get_arg_val<uint32_t>(0);

    // every CB is fp32; the UnpackToDestFp32 / Default split is reconfigured per op group
    compute_kernel_hw_startup(cb_xh, cb_xh, cb_out);

    CircularBuffer q(cb_q), k(cb_k), v(cb_v), decay(cb_decay), beta(cb_beta), S(cb_S);
    CircularBuffer qnh(cb_qnh), qnl(cb_qnl), knh(cb_knh), knl(cb_knl), dcol(cb_dcol), sdx(cb_sdx), sdh(cb_sdh);
    CircularBuffer sdl(cb_sdl), vread(cb_vread), qsd(cb_qsd), qk(cb_qk), uh(cb_uh), ul(cb_ul), kcol(cb_kcol);
    CircularBuffer kcoll(cb_kcoll), kn1h(cb_kn1h), kn1l(cb_kn1l);

    for (uint32_t w = 0; w < count; w++) {
        // L2 norms (q also scaled), split for the matmuls
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

        // sd = diag(decay) S (exact: the decay row is transposed in DST, the product on the SFPU)
        decay.wait_front(Kt);
        transpose_block(cb_decay, cb_dcol, Kt);
        decay.pop_front(Kt);
        dcol.wait_front(Kt);
        S.wait_front(kv);
        decay_state<Kt, Vt>();
        S.pop_front(kv);
        dcol.pop_front(Kt);
        sdh.wait_front(kv);
        sdl.wait_front(kv);

        // vread = kn . sd, qsd = qn . sd, qk = qn . kn
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

        // u = beta * (v - vread) ; o = qsd + qk * u
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

        // S_new = sd + kn^T (x) u
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
    }
}
