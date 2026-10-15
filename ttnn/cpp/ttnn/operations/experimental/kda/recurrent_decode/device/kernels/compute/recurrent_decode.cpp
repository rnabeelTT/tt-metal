// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// One KDA decode step per head (derived from the compute kernel of ttnn.transformer.fused_recurrent_gated_delta_rule;
// here the decay is per key channel and the q / k L2 norms run in-kernel):
//   qn    = q * q_scale / sqrt(q.q + eps),  kn = k / sqrt(k.k + eps)
//   sd    = diag(decay) S                 (decay row transposed to columns, column-broadcast multiply)
//   vread = kn . sd                       ([1,V])
//   u     = beta * (v - vread)            ([1,V])
//   S_new = sd + kn^T (x) u               (rank-1 update accumulated onto sd in DST)
//   o     = qn . S_new
// Head rows are [1, D] tiles (row 0 valid, rows 1..31 zero), so x . x^T lands the squared norm in element [0,0] and
// the inner-dim-1 outer product is a matmul whose 31 padding lanes are zero.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/dataflow/circular_buffer.h"

namespace {

constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_decay = 3, cb_beta = 4, cb_S = 5, cb_out = 6, cb_state = 7;
constexpr uint32_t cb_snew = 8, cb_sd = 9, cb_vread = 10, cb_u = 11, cb_kcol = 12, cb_dcol = 13, cb_qn = 14, cb_kn = 15;
constexpr uint32_t cb_nrm = 16;

constexpr uint32_t DST_TILES = 4;  // fp32 DST tiles per tile_regs_acquire (half-sync)

// out[Mt,Nt] = A[Mt,Kt] @ B[Kt,Nt]. Inputs must already be available.
void mm(uint32_t a, uint32_t b, uint32_t o, uint32_t Mt, uint32_t Kt, uint32_t Nt) {
    CircularBuffer cb_o(o);
    cb_o.reserve_back(Mt * Nt);
    matmul_init(a, b, 0);
    for (uint32_t mi = 0; mi < Mt; mi++) {
        for (uint32_t ni = 0; ni < Nt; ni++) {
            tile_regs_acquire();
            for (uint32_t ki = 0; ki < Kt; ki++) {
                matmul_tiles(a, b, mi * Kt + ki, ki * Nt + ni, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, o, mi * Nt + ni);
            tile_regs_release();
        }
    }
    cb_o.push_back(Mt * Nt);
}

// nrm[0,0] = scale / sqrt(sum_i x_i . x_i^T + eps) over the n row tiles of x (x must be available).
void inv_norm(uint32_t x, uint32_t n, uint32_t eps_bits, uint32_t scale_bits, bool apply_scale) {
    CircularBuffer cb_o(cb_nrm);
    cb_o.reserve_back(1);
    tile_regs_acquire();
    matmul_init(x, x, 1);
    for (uint32_t i = 0; i < n; i++) {
        matmul_tiles(x, x, i, i, 0);
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
    cb_o.push_back(1);
}

// out = A * scalar, n tiles; scalar = the [0,0] element of the single `scal` tile.
void bcast_scalar_mul(uint32_t a, uint32_t scal, uint32_t o, uint32_t n) {
    CircularBuffer cb_o(o);
    cb_o.reserve_back(n);
    mul_bcast_scalar_init(a, scal);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        mul_tiles_bcast_scalar(a, scal, i, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_o.push_back(n);
}

// sd[i, j] = S[i, j] * dcol[i] broadcast along the columns (row k of the state scaled by decay[k]).
template <uint32_t Kt, uint32_t Vt>
void decay_state() {
    CircularBuffer cb_o(cb_sd);
    cb_o.reserve_back(Kt * Vt);
    mul_bcast_cols_init(cb_S, cb_dcol);
    for (uint32_t i = 0; i < Kt; i++) {
        for (uint32_t j0 = 0; j0 < Vt; j0 += DST_TILES) {
            const uint32_t nn = (Vt - j0 < DST_TILES) ? (Vt - j0) : DST_TILES;
            tile_regs_acquire();
            for (uint32_t j = 0; j < nn; j++) {
                mul_tiles_bcast_cols(cb_S, cb_dcol, i * Vt + j0 + j, i, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < nn; j++) {
                pack_tile<true>(j, cb_sd, i * Vt + j0 + j);
            }
            tile_regs_release();
        }
    }
    cb_o.push_back(Kt * Vt);
}

// S_new[Kt,Vt] = sd + kcol[Kt,1] (x) u[1,Vt]: each DST tile is seeded with sd and the rank-1 matmul accumulates onto
// it; the result is packed to cb_snew (read back for o) and cb_state (written to DRAM).
template <uint32_t Kt, uint32_t Vt>
void rank1_update() {
    constexpr uint32_t kv = Kt * Vt;
    CircularBuffer cb_o(cb_snew), cb_st(cb_state);
    cb_o.reserve_back(kv);
    cb_st.reserve_back(kv);
    for (uint32_t mi = 0; mi < Kt; mi++) {
        for (uint32_t n0 = 0; n0 < Vt; n0 += DST_TILES) {
            const uint32_t nn = (Vt - n0 < DST_TILES) ? (Vt - n0) : DST_TILES;
            tile_regs_acquire();
            copy_init(cb_sd);
            for (uint32_t j = 0; j < nn; j++) {
                copy_tile(cb_sd, mi * Vt + n0 + j, j);
            }
            matmul_init(cb_kcol, cb_u, 0);
            for (uint32_t j = 0; j < nn; j++) {
                matmul_tiles(cb_kcol, cb_u, mi, n0 + j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < nn; j++) {
                pack_tile<true>(j, cb_snew, mi * Vt + n0 + j);
                pack_tile<true>(j, cb_state, mi * Vt + n0 + j);
            }
            tile_regs_release();
        }
    }
    cb_o.push_back(kv);
    cb_st.push_back(kv);
}

// out = (A - B) * beta, n tiles (beta at [0,0] of its tile, rows 1..31 zero; only row 0 of A / B carries data).
void delta_rule_residual(uint32_t a, uint32_t b, uint32_t beta, uint32_t o, uint32_t n) {
    constexpr uint32_t per_batch = DST_TILES - 1;  // the last DST slot holds beta
    CircularBuffer cb_o(o);
    cb_o.reserve_back(n);
    for (uint32_t i0 = 0; i0 < n; i0 += per_batch) {
        const uint32_t nn = (n - i0 < per_batch) ? (n - i0) : per_batch;
        tile_regs_acquire();
        sub_init(a, b);
        for (uint32_t j = 0; j < nn; j++) {
            sub_tiles(a, b, i0 + j, i0 + j, j);
        }
        copy_init(beta);
        copy_tile(beta, 0, nn);
        sfpu_mul_bcast_col_init();
        for (uint32_t j = 0; j < nn; j++) {
            sfpu_mul_bcast_col(j, nn);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < nn; j++) {
            pack_tile<true>(j, o, i0 + j);
        }
        tile_regs_release();
    }
    cb_o.push_back(n);
}

// out[Kt] = per-tile transpose of in[Kt] (a [1,K] row becomes a [K,1] column). in must be available.
void transpose_block(uint32_t in, uint32_t o, uint32_t n) {
    CircularBuffer cb_o(o);
    cb_o.reserve_back(n);
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

}  // namespace

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(2);
    constexpr uint32_t q_scale_bits = get_compile_time_arg_val(3);
    constexpr uint32_t kv = Kt * Vt;
    const uint32_t count = get_arg_val<uint32_t>(0);

    // every CB is fp32, so the formats configured here hold for every op below
    compute_kernel_hw_startup(cb_q, cb_v, cb_out);

    CircularBuffer q(cb_q), k(cb_k), v(cb_v), decay(cb_decay), beta(cb_beta), S(cb_S);
    CircularBuffer qn(cb_qn), kn(cb_kn), nrm(cb_nrm), dcol(cb_dcol), sd(cb_sd), vread(cb_vread), u(cb_u);
    CircularBuffer kcol(cb_kcol), snew(cb_snew);

    for (uint32_t w = 0; w < count; w++) {
        // L2 norms (q also scaled)
        q.wait_front(Kt);
        inv_norm(cb_q, Kt, eps_bits, q_scale_bits, true);
        nrm.wait_front(1);
        bcast_scalar_mul(cb_q, cb_nrm, cb_qn, Kt);
        nrm.pop_front(1);
        q.pop_front(Kt);
        k.wait_front(Kt);
        inv_norm(cb_k, Kt, eps_bits, q_scale_bits, false);
        nrm.wait_front(1);
        bcast_scalar_mul(cb_k, cb_nrm, cb_kn, Kt);
        nrm.pop_front(1);
        k.pop_front(Kt);
        kn.wait_front(Kt);

        // sd = diag(decay) S
        decay.wait_front(Kt);
        transpose_block(cb_decay, cb_dcol, Kt);
        decay.pop_front(Kt);
        dcol.wait_front(Kt);
        S.wait_front(kv);
        decay_state<Kt, Vt>();
        S.pop_front(kv);
        dcol.pop_front(Kt);
        sd.wait_front(kv);

        // vread = kn . sd ; u = beta * (v - vread)
        mm(cb_kn, cb_sd, cb_vread, 1, Kt, Vt);
        vread.wait_front(Vt);
        v.wait_front(Vt);
        beta.wait_front(1);
        delta_rule_residual(cb_v, cb_vread, cb_beta, cb_u, Vt);
        v.pop_front(Vt);
        vread.pop_front(Vt);
        beta.pop_front(1);
        u.wait_front(Vt);

        // S_new = sd + kn^T (x) u
        transpose_block(cb_kn, cb_kcol, Kt);
        kn.pop_front(Kt);
        kcol.wait_front(Kt);
        rank1_update<Kt, Vt>();
        kcol.pop_front(Kt);
        u.pop_front(Vt);
        sd.pop_front(kv);

        // o = qn . S_new
        snew.wait_front(kv);
        qn.wait_front(Kt);
        mm(cb_qn, cb_snew, cb_out, 1, Kt, Vt);
        qn.pop_front(Kt);
        snew.pop_front(kv);
    }
}
