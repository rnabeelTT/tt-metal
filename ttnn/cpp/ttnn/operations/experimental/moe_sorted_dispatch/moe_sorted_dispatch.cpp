// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/moe_sorted_dispatch/moe_sorted_dispatch.hpp"

#include "device/moe_sorted_dispatch_device_operation.hpp"

namespace ttnn::operations::experimental {

std::vector<ttnn::Tensor> moe_sorted_dispatch(
    const Tensor& ids,
    const Tensor& scores,
    uint32_t num_experts,
    uint32_t top_k,
    uint32_t group,
    uint32_t dispatched_rows,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    uint32_t pad_group_id) {
    return ttnn::prim::moe_sorted_dispatch(
        ids, scores, num_experts, top_k, group, dispatched_rows, memory_config, pad_group_id);
}

}  // namespace ttnn::operations::experimental
