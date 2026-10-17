// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused mHC parametrization on one core. Per token-tile (up to 32 tokens along tile rows):
//   comb = Sinkhorn( exp( mixes @ SEL_comb + base_comb ) )   via permutation-sum row / col sums
//   pre  = sigmoid( mixes @ SEL_pre  + base_pre  ) + eps
//   post = 2 * sigmoid( mixes @ SEL_post + base_post )
// SEL_* extract+left-align each group from the 24-wide mixes and fold in the scalar a; the
// Sinkhorn normalizations are same-shape `m / rowsum(m)` / `m / colsum(m)` divides, the sums broadcast
// back to every entry of the group (sum_s m @ P_s), so the whole op is matmul + SFPU tiles. (The RB / CB
// sum matrices in consts are no longer read: their 4-term in-face sums are not fp32-exact on the FPU.)
//
// fp32 precision: the FPU reads fp32 at TF32 (10-bit mantissa, truncating), and its multiplier uses all 11
// significant bits of srcB (matmul in0) but only the top 10 of srcA (matmul in1). So no fp32 value is fed to it whole.
// Elementwise work (bias add, exp / sigmoid / reciprocal, the normalizing multiply) runs on the SFPU on DST tiles
// unpacked straight to DST (mixes, consts and m are UnpackToDestFp32 CBs, see the program factory). The in0 operands
// (mixes, m) are split x = hi + lo in DST (hi = 11 significant bits, lo = x - hi exact); the in1 constants into three
// 10-bit pieces (exact in srcA). The FPU also sums the products inside a 16-wide face at reduced precision (~2^-12 of
// the largest), so every matmul here has one nonzero product per output: the projections are selections, and the
// Sinkhorn row / column sums add n permuted copies of m (m @ P_s, reader-made permutation tiles) across separate
// matmul calls, which accumulate in fp32 DST. ~1e-7 relative instead of ~1e-3.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

#ifdef TRISC_MATH
namespace ckernel::sfpu {
// DST[hi] = DST[x] with the low mantissa bits cleared (MASK), DST[lo] = DST[x] - DST[hi] (exact). hi may alias x.
template <std::uint32_t MASK, int ITERATIONS = 8>
inline void calculate_mhc_split(const std::uint32_t x, const std::uint32_t hi, const std::uint32_t lo) {
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

template <uint32_t MASK = IN0_MASK>
ALWI void split_tile(uint32_t x, uint32_t hi, uint32_t lo) {
    MATH((SFPU_BINARY_INIT(unused)));
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_mhc_split, (MASK, 8), x, hi, lo, VectorMode::RC)));
}

constexpr uint32_t CB_MIXES = 0;   // c_0  (mixes, streamed; UnpackToDestFp32)
constexpr uint32_t CB_CONSTS = 1;  // c_1  (8 resident tiles; UnpackToDestFp32)
constexpr uint32_t CB_PRE = 2;     // c_2  out
constexpr uint32_t CB_POST = 3;    // c_3  out
constexpr uint32_t CB_COMB = 4;    // c_4  out
constexpr uint32_t CB_PERM = 5;    // c_5  2n-1 permutation tiles: identity, n-1 row shifts, n-1 column shifts
constexpr uint32_t CB_M = 16;      // m, exact (UnpackToDestFp32)
constexpr uint32_t CB_MH = 17;     // m, hi (matmul operand)
constexpr uint32_t CB_ML = 18;     // m, lo (matmul operand)
constexpr uint32_t CB_XH = 24;     // mixes, hi
constexpr uint32_t CB_XL = 25;     // mixes, lo
constexpr uint32_t CB_CH = 26;     // consts, top 10 significant bits (8 resident tiles)
constexpr uint32_t CB_CM = 27;     // consts, next 10 bits
constexpr uint32_t CB_CL = 28;     // consts, the rest (<= 4 bits)

// const-tile indices within CB_CONSTS / CB_CH / CB_CL
constexpr uint32_t SEL_PRE = 0, SEL_POST = 1, SEL_COMB = 2;
constexpr uint32_t BASE_PRE = 3, BASE_POST = 4, BASE_COMB = 5;
constexpr uint32_t NUM_CONSTS = 8;

constexpr uint32_t TWO_BITS = 0x40000000u;  // 2.0f

ALWI void copy_from(uint32_t cb) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
}

ALWI void mm_begin(uint32_t a, uint32_t b) {
    reconfig_data_format(a, b);
    matmul_init(a, b);
}

// DST[0] = mixes @ SEL[sel] + BASE[base] (fp32-exact up to ~1e-7). XH / XL must be available; pops nothing.
ALWI void project(uint32_t sel, uint32_t base) {
    mm_begin(CB_XH, CB_CH);
    // the a folded into SEL is arbitrary fp32: all three pieces against hi, the first two against lo
    matmul_tiles(CB_XH, CB_CH, 0, sel, 0);
    matmul_tiles(CB_XH, CB_CM, 0, sel, 0);
    matmul_tiles(CB_XH, CB_CL, 0, sel, 0);
    matmul_tiles(CB_XL, CB_CH, 0, sel, 0);
    matmul_tiles(CB_XL, CB_CM, 0, sel, 0);
    copy_from(CB_CONSTS);
    copy_tile(CB_CONSTS, base, 1);
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
}

// pack DST[x] (exact) to CB_M and its split halves to CB_MH / CB_ML (uses DST[x+1], DST[x+2])
ALWI void pack_m(uint32_t x) {
    split_tile(x, x + 1, x + 2);
    tile_regs_commit();
    CircularBuffer cm(CB_M), ch(CB_MH), cl(CB_ML);
    cm.reserve_back(1);
    ch.reserve_back(1);
    cl.reserve_back(1);
    tile_regs_wait();
    pack_tile(x, CB_M);
    pack_tile(x + 1, CB_MH);
    pack_tile(x + 2, CB_ML);
    tile_regs_release();
    cm.push_back(1);
    ch.push_back(1);
    cl.push_back(1);
}

// m <- m / (row or column sums of m [+ eps]) [+ post_eps]. Consumes the current m (M / MH / ML); the result goes back
// to M / MH / ML, or to CB_COMB when final.
template <uint32_t n>
FORCE_INLINE void normalize(bool rows, bool denom_eps, bool post_eps, uint32_t eps_bits, bool final) {
    CircularBuffer cm(CB_M), ch(CB_MH), cl(CB_ML);
    cm.wait_front(1);
    ch.wait_front(1);
    cl.wait_front(1);
    tile_regs_acquire();
    mm_begin(CB_MH, CB_PERM);
    for (uint32_t s = 0; s < n; ++s) {  // sum over the group = sum of n permuted copies
        const uint32_t perm = s == 0 ? 0 : (rows ? s : n - 1 + s);
        matmul_tiles(CB_MH, CB_PERM, 0, perm, 0);
        matmul_tiles(CB_ML, CB_PERM, 0, perm, 0);
    }
    if (denom_eps) {
        binop_with_scalar_tile_init();
        add_unary_tile(0, eps_bits);
    }
    // non-legacy reciprocal: fp32-accurate (the legacy_compat default runs a fixed 3-step Newton from a linear guess,
    // ~1.4e-3 low for some inputs, which biases every Sinkhorn normalization)
    recip_tile_init<false>();
    recip_tile<false>(0);
    copy_from(CB_M);
    copy_tile(CB_M, 0, 1);
    mul_binary_tile_init();
    mul_binary_tile(1, 0, 1);
    if (post_eps) {
        binop_with_scalar_tile_init();
        add_unary_tile(1, eps_bits);
    }
    cm.pop_front(1);
    ch.pop_front(1);
    cl.pop_front(1);
    if (final) {
        tile_regs_commit();
        CircularBuffer co(CB_COMB);
        co.reserve_back(1);
        tile_regs_wait();
        pack_tile(1, CB_COMB);
        tile_regs_release();
        co.push_back(1);
    } else {
        pack_m(1);
    }
}

}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(CB_MIXES, CB_CONSTS, CB_COMB);  // must precede any other compute work
    const uint32_t num_token_tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t iters = get_compile_time_arg_val(0);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(1);
    constexpr uint32_t n = get_compile_time_arg_val(2);

    CircularBuffer cb_consts(CB_CONSTS), cb_mixes(CB_MIXES), cb_ch(CB_CH), cb_cm(CB_CM), cb_cl(CB_CL);
    CircularBuffer cb_xh(CB_XH), cb_xl(CB_XL);
    CircularBuffer cb_pre(CB_PRE), cb_post(CB_POST), cb_perm(CB_PERM);
    cb_consts.wait_front(NUM_CONSTS);  // constants resident for the whole op
    cb_perm.wait_front(2 * n - 1);

    // split the constants once (the a scalars in SEL_* are arbitrary fp32) into three srcA-exact pieces
    cb_ch.reserve_back(NUM_CONSTS);
    cb_cm.reserve_back(NUM_CONSTS);
    cb_cl.reserve_back(NUM_CONSTS);
    for (uint32_t i = 0; i < NUM_CONSTS; ++i) {
        tile_regs_acquire();
        copy_from(CB_CONSTS);
        copy_tile(CB_CONSTS, i, 0);
        split_tile<IN1_MASK>(0, 1, 2);  // 1 = top 10 bits, 2 = rest
        split_tile<IN1_MASK>(2, 2, 3);  // 2 = next 10 bits, 3 = rest
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(1, CB_CH, i);
        pack_tile<true>(2, CB_CM, i);
        pack_tile<true>(3, CB_CL, i);
        tile_regs_release();
    }
    cb_ch.push_back(NUM_CONSTS);
    cb_cm.push_back(NUM_CONSTS);
    cb_cl.push_back(NUM_CONSTS);
    cb_ch.wait_front(NUM_CONSTS);
    cb_cm.wait_front(NUM_CONSTS);
    cb_cl.wait_front(NUM_CONSTS);

    for (uint32_t t = 0; t < num_token_tiles; ++t) {
        cb_mixes.wait_front(1);
        cb_xh.reserve_back(1);
        cb_xl.reserve_back(1);
        tile_regs_acquire();
        copy_from(CB_MIXES);
        copy_tile(CB_MIXES, 0, 0);
        split_tile(0, 1, 2);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(1, CB_XH);
        pack_tile(2, CB_XL);
        tile_regs_release();
        cb_xh.push_back(1);
        cb_xl.push_back(1);
        cb_mixes.pop_front(1);
        cb_xh.wait_front(1);
        cb_xl.wait_front(1);

        // ---- comb = Sinkhorn(exp(mixes @ SEL_comb + base_comb)) ----
        tile_regs_acquire();
        project(SEL_COMB, BASE_COMB);
        // Cap logits before exp so an out-of-range learned logit cannot overflow to inf
        // (which becomes inf/inf = NaN in the softmax divide). The softmax ratio is unchanged
        // for any realistic logit (a_res init ~ 0.01 => |logit| << 1) -- equivalent to the
        // canonical row-max subtraction within range, with no cross-lane reduce.
        unary_min_tile_init();
        unary_min_tile(0, 0x42a00000u);  // min(x, 80.0f)
        exp_tile_init();
        exp_tile(0);
        pack_m(0);
        normalize<n>(true, false, true, eps_bits, false);        // row softmax, + eps
        normalize<n>(false, true, false, eps_bits, iters == 1);  // first column norm
        for (uint32_t i = 1; i < iters; ++i) {
            normalize<n>(true, true, false, eps_bits, false);            // row
            normalize<n>(false, true, false, eps_bits, i + 1 == iters);  // col
        }

        // ---- pre = sigmoid(mixes @ SEL_pre + base_pre) + eps ----
        tile_regs_acquire();
        project(SEL_PRE, BASE_PRE);
        sigmoid_tile_init();
        sigmoid_tile(0);
        binop_with_scalar_tile_init();
        add_unary_tile(0, eps_bits);
        tile_regs_commit();
        cb_pre.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, CB_PRE);
        tile_regs_release();
        cb_pre.push_back(1);

        // ---- post = 2 * sigmoid(mixes @ SEL_post + base_post) ----
        tile_regs_acquire();
        project(SEL_POST, BASE_POST);
        sigmoid_tile_init();
        sigmoid_tile(0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, TWO_BITS);
        tile_regs_commit();
        cb_post.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, CB_POST);
        tile_regs_release();
        cb_post.push_back(1);

        cb_xh.pop_front(1);
        cb_xl.pop_front(1);
    }
}
