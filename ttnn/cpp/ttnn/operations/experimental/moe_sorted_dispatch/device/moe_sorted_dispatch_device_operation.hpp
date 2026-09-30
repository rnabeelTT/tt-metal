// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "moe_sorted_dispatch_device_operation_types.hpp"
#include "moe_sorted_dispatch_program_factory.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct MoeSortedDispatchDeviceOperation {
    using operation_attributes_t = MoeSortedDispatchParams;
    using tensor_args_t = MoeSortedDispatchInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<MoeSortedDispatchProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
// Returns {src_tok uint32 [1, Rd], weights bfloat16 [1, Rd], group_ids uint16 [1, G], undispatch uint32 [1, T*k]}.
std::vector<Tensor> moe_sorted_dispatch(
    const Tensor& ids,
    const Tensor& scores,
    uint32_t num_experts,
    uint32_t top_k,
    uint32_t group,
    uint32_t dispatched_rows,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    uint32_t pad_group_id = 0);
}  // namespace ttnn::prim
