// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/moe_gather_add_rows/moe_gather_add_rows.hpp"

#include "device/moe_gather_add_rows_device_operation.hpp"

namespace ttnn::operations::experimental {

ttnn::Tensor moe_gather_add_rows(
    const Tensor& rows,
    const Tensor& undispatch,
    uint32_t num_tokens,
    uint32_t top_k,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config) {
    return ttnn::prim::moe_gather_add_rows(rows, undispatch, num_tokens, top_k, memory_config);
}

}  // namespace ttnn::operations::experimental
