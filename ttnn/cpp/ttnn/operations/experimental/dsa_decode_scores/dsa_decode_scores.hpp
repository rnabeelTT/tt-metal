// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental {

// Lightning-indexer scores of one decode query per slot over the slot's PAGED pooled-key cache (GLM-5 / DSA):
//
//     score[s, t] = sum_h relu(q[s, h, :] . k_s[t, :]) * w[s, h]     for key t < limits[s], else -inf
//
// where k_s[t] = k_pool.row(page_table[s][t / 32] * 32 + t % 32) (32-key page blocks, one tile row each).
//
//   q:          [1, 1, S*32, D] BFLOAT16 TILE  (row s*32 + h: query head h of slot s; 32 heads)
//   weights:    [1, 1, S, 32] FLOAT32 TILE     (row s: per-head weights of slot s, scales folded in)
//   k_pool:     [.., .., T, D] BFLOAT16 TILE   (32-key tile rows; pages tile_row * D/32 + d)
//   page_table: [rows >= S, max_blocks] INT32 / UINT32 ROW_MAJOR
//   limits:     [1, S] UINT32 ROW_MAJOR        (valid keys of slot s; blocks starting at or past it are not read)
//   num_blocks: scored width in 32-key blocks.
//
// Returns scores [1, 1, S, num_blocks * 32] FLOAT32 TILE. Columns of a partially valid block past limits[s] hold
// scores of whatever keys sit there; callers mask t >= limits[s].
ttnn::Tensor dsa_decode_scores(
    const Tensor& q,
    const Tensor& weights,
    const Tensor& k_pool,
    const Tensor& page_table,
    const Tensor& limits,
    uint32_t num_blocks);

}  // namespace ttnn::operations::experimental
