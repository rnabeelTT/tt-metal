// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::mhc_pre {

// Fused mHC pre-collapse: the stream reduction that feeds the wrapped sub-block,
//
//     h = sum_i pre[:, i] * residual[:, i*C:(i+1)*C]                 (packed [1,1,T,n*C] layout)
//
// i.e. what n column slices of residual, n single-column slices of pre and n multiply / addcmul ops compute,
// in one pass that reads each (token-tile, column-tile) of the n streams once.
//
//   residual: [1,1,T,n*C]  FLOAT32 / BFLOAT16 TILE  -- the n residual streams, stream i at [i*C,(i+1)*C)
//   pre:      [1,1,T,n]    same dtype, TILE         -- mhc_split_sinkhorn's pre output
//   consts:   [n*n,32,32]  same dtype, TILE         -- the column-broadcast tiles of mhc_post (tile k: row k ones)
//   output_dtype: dtype of h (default: the input dtype); the sum is accumulated in fp32 DEST either way.
//
// Returns h [1,1,T,C] TILE.
ttnn::Tensor mhc_pre(
    const ttnn::Tensor& residual,
    const ttnn::Tensor& pre,
    const ttnn::Tensor& consts,
    uint32_t n,
    std::optional<tt::tt_metal::DataType> output_dtype = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::mhc_pre

namespace ttnn {
using operations::experimental::deepseek_prefill::mhc_pre::mhc_pre;
}  // namespace ttnn
