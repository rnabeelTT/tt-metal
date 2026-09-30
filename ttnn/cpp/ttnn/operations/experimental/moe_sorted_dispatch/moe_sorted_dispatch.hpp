// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental {

// Expert-sorted MoE dispatch bookkeeping in one single-core program. ids / scores: ROW_MAJOR [1, T*k] (uint32 /
// bfloat16, token-major pairs). Returns (src_tok uint32 [1, Rd], weights bfloat16 [1, Rd], group_ids uint16 [1, G],
// undispatch uint32 [1, T*k]) with Rd = dispatched_rows and G = Rd / group.
std::vector<ttnn::Tensor> moe_sorted_dispatch(
    const Tensor& ids,
    const Tensor& scores,
    uint32_t num_experts,
    uint32_t top_k,
    uint32_t group,
    uint32_t dispatched_rows,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    uint32_t pad_group_id = 0);

}  // namespace ttnn::operations::experimental
