// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental {

// MoE dispatch gather: out[g, r, :] = rows[src_tok[g*32 + r], :] for every REAL group g (group_ids[g] < num_experts),
// written straight as tiles [1, G, 32, H]; padding groups (id >= num_experts) are skipped (their rows are never read
// by the grouped sparse matmul that skips padding groups) - their output tiles hold whatever the buffer held.
ttnn::Tensor moe_gather_rows(
    const Tensor& rows,
    const Tensor& src_tok,
    const Tensor& group_ids,
    uint32_t num_experts,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::operations::experimental
