// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental {

// MoE un-dispatch: out[t, :] = sum_{j < k} rows[undispatch[t*k + j], :]  (the k expert outputs of every token).
// rows: bfloat16 ROW_MAJOR [Rd, H] interleaved (one stick per dispatched row); undispatch: uint32 ROW_MAJOR [1, T*k].
// Returns bfloat16 TILE [1, 1, T, H]. Multi-core: one work item per (32-token tile, column block).
ttnn::Tensor moe_gather_add_rows(
    const Tensor& rows,
    const Tensor& undispatch,
    uint32_t num_tokens,
    uint32_t top_k,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::operations::experimental
