// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/dsa_decode_scores/dsa_decode_scores.hpp"
#include "ttnn/operations/experimental/dsa_decode_scores/device/dsa_decode_scores_device_operation.hpp"

namespace ttnn::operations::experimental {

ttnn::Tensor dsa_decode_scores(
    const Tensor& q,
    const Tensor& weights,
    const Tensor& k_pool,
    const Tensor& page_table,
    const Tensor& limits,
    uint32_t num_blocks) {
    return ttnn::prim::dsa_decode_scores(q, weights, k_pool, page_table, limits, num_blocks);
}

}  // namespace ttnn::operations::experimental
