// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/operations/experimental/moe_gather_rows/moe_gather_rows.hpp"

#include "device/moe_gather_rows_device_operation.hpp"

namespace ttnn::operations::experimental {

ttnn::Tensor moe_gather_rows(
    const Tensor& rows,
    const Tensor& src_tok,
    const Tensor& group_ids,
    uint32_t num_experts,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config) {
    return ttnn::prim::moe_gather_rows(rows, src_tok, group_ids, num_experts, memory_config);
}

}  // namespace ttnn::operations::experimental
