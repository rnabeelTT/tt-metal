// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::mhc_mix {

// Split-K partials of the mHC mix projection and of the row sums of squares (both over the packed n*C-wide stream):
//
//     partials[0, c, :, 0:32]  = x[:, chunk c] @ fn_t[chunk c, :]          (projection partial)
//     partials[0, c, :, 32:64] = x[:, chunk c] @ x[:, chunk c]^T  per 32-row tile (Gram partial; its diagonal is the
//                                                                  chunk's row sum of squares)
//
// over num_chunks equal K chunks, one (row tile, chunk) unit per core, instead of one single-core matmul and one
// single-core mean over the 16384-wide row. The caller sums the chunks and applies rsqrt(mean(x^2) + eps).
//
//   x:     [1, 1, T, K]  FLOAT32 TILE
//   fn_t:  [K, 32]       FLOAT32 TILE (the projection, zero padded to 32 columns)
// Returns partials [1, num_chunks, T, 64] FLOAT32 TILE.
ttnn::Tensor mhc_mix(const ttnn::Tensor& x, const ttnn::Tensor& fn_t, uint32_t num_chunks);

}  // namespace ttnn::operations::experimental::deepseek_prefill::mhc_mix

namespace ttnn {
using operations::experimental::deepseek_prefill::mhc_mix::mhc_mix;
}  // namespace ttnn
