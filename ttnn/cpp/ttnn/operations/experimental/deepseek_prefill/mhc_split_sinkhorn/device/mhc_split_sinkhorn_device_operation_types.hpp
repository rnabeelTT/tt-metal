// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct MhcSplitSinkhornParams {
    uint32_t n;  // expansion rate (streams); mix_hc = (2+n)*n
    uint32_t sinkhorn_iters;
    float eps;
    uint32_t max_cores;  // 0 = full grid; 1 pins single-core (benchmark A/B). Hashed for cache correctness.
    // Math fidelity of the Sinkhorn row / column-sum matmuls (m_hi / m_lo @ permutation tile). 4 = HiFi4; 3 = HiFi3,
    // which skips only the srcA-low x srcB-low phase: the permutation operand (srcA) is 0 / 1, so that phase adds exact
    // zeros.
    uint32_t perm_fidelity;
    // Face-restricted Sinkhorn (MHC_SINKHORN_FACES=1, n*n <= 16): the 40 normalizations run their SFPU work only on the
    // DST faces that hold the n*n used columns: 1 = face 0 (one token-tile, T <= 16 rows), 2 = faces 0 and 2 (columns
    // 0-15, all rows); 0 = all 4 faces. The padding of m is zeroed once after exp, so it stays exactly 0 through every
    // permutation matmul; comb's padding comes out 0 (it was ~1 before). Valid entries are bit-identical.
    uint32_t faces;
};

// mixes: fused-projection output [T, (2+n)*n].  consts: [8, 32, 32] host-prepared tiles
// (SEL_pre, SEL_post, SEL_comb with a folded in; base_pre/post/comb; RB, CB) -- see the
// Python wrapper. The scalars a and biases b are baked into these tiles host-side.
struct MhcSplitSinkhornTensorArgs {
    const Tensor& mixes;
    const Tensor& consts;
};

}  // namespace ttnn::experimental::prim
