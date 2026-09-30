// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_sorted_dispatch_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

void MoeSortedDispatchDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& a, const tensor_args_t& t) {
    const uint32_t R = a.num_tokens * a.top_k;
    TT_FATAL(
        t.ids.storage_type() == StorageType::DEVICE && t.scores.storage_type() == StorageType::DEVICE,
        "moe_sorted_dispatch: operands must be on device");
    TT_FATAL(
        t.ids.dtype() == DataType::UINT16 || t.ids.dtype() == DataType::UINT32 || t.ids.dtype() == DataType::INT32,
        "moe_sorted_dispatch: ids must be uint16 / uint32 / int32, got {}",
        t.ids.dtype());
    TT_FATAL(
        t.scores.dtype() == DataType::BFLOAT16,
        "moe_sorted_dispatch: scores must be bfloat16, got {}",
        t.scores.dtype());
    TT_FATAL(
        t.ids.layout() == Layout::TILE && t.scores.layout() == Layout::TILE,
        "moe_sorted_dispatch: ids / scores must be TILE layout [.., T, k] (read straight from the top-k output)");
    TT_FATAL(!t.ids.is_sharded() && !t.scores.is_sharded(), "moe_sorted_dispatch: interleaved operands only");
    TT_FATAL(
        t.ids.logical_volume() == R && t.scores.logical_volume() == R,
        "moe_sorted_dispatch: ids / scores must hold T*k = {} elements, got {} / {}",
        R,
        t.ids.logical_volume(),
        t.scores.logical_volume());
    TT_FATAL(
        t.ids.logical_shape()[-1] == a.top_k && t.scores.logical_shape()[-1] == a.top_k &&
            t.ids.logical_shape()[-2] == a.num_tokens && t.scores.logical_shape()[-2] == a.num_tokens,
        "moe_sorted_dispatch: ids / scores must be [.., T={}, k={}]",
        a.num_tokens,
        a.top_k);
    TT_FATAL(a.top_k <= 32, "moe_sorted_dispatch: top_k must fit one tile column block (<= 32), got {}", a.top_k);
    TT_FATAL(
        t.ids.tensor_spec().tile().get_height() == 32 && t.ids.tensor_spec().tile().get_width() == 32 &&
            t.scores.tensor_spec().tile().get_height() == 32 && t.scores.tensor_spec().tile().get_width() == 32,
        "moe_sorted_dispatch: 32x32 tiles only");
    TT_FATAL(a.group % 32 == 0 && a.group > 0, "moe_sorted_dispatch: group must be a positive multiple of 32");
    TT_FATAL(
        a.num_groups * a.group == a.dispatched_rows, "moe_sorted_dispatch: dispatched_rows must be num_groups * group");
    // worst case: every expert's last group padded + the trailing padding group must fit the static bound
    const uint32_t worst = R + ((a.group - 1) * a.num_experts + a.group - 1) / a.group * a.group + a.group;
    TT_FATAL(
        a.dispatched_rows >= worst || a.dispatched_rows >= R + a.num_experts * a.group,
        "moe_sorted_dispatch: dispatched_rows {} below the worst-case padded size {}",
        a.dispatched_rows,
        worst);
}

void MoeSortedDispatchDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& a, const tensor_args_t& t) {
    validate_on_program_cache_miss(a, t);
}

MoeSortedDispatchDeviceOperation::spec_return_value_t MoeSortedDispatchDeviceOperation::compute_output_specs(
    const operation_attributes_t& a, const tensor_args_t&) {
    const uint32_t R = a.num_tokens * a.top_k;
    auto spec = [&](uint32_t n, DataType dt) {
        return TensorSpec(ttnn::Shape({1, n}), TensorLayout(dt, PageConfig(Layout::ROW_MAJOR), a.output_mem_config));
    };
    return {
        spec(a.dispatched_rows, DataType::UINT32),    // src_tok
        spec(a.dispatched_rows, DataType::BFLOAT16),  // weights
        spec(a.num_groups, DataType::UINT16),         // group_ids
        spec(R, DataType::UINT32),                    // undispatch
    };
}

MoeSortedDispatchDeviceOperation::tensor_return_value_t MoeSortedDispatchDeviceOperation::create_output_tensors(
    const operation_attributes_t& a, const tensor_args_t& t) {
    std::vector<Tensor> out;
    for (const auto& s : compute_output_specs(a, t)) {
        out.push_back(create_device_tensor(s, t.ids.device()));
    }
    return out;
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<Tensor> moe_sorted_dispatch(
    const Tensor& ids,
    const Tensor& scores,
    uint32_t num_experts,
    uint32_t top_k,
    uint32_t group,
    uint32_t dispatched_rows,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    uint32_t pad_group_id) {
    using Op = ttnn::experimental::prim::MoeSortedDispatchDeviceOperation;
    const uint32_t R = ids.logical_volume();
    TT_FATAL(top_k > 0 && R % top_k == 0, "moe_sorted_dispatch: ids volume {} not a multiple of top_k {}", R, top_k);
    TT_FATAL(
        ids.logical_shape().rank() >= 2 && ids.logical_shape()[-1] == top_k,
        "moe_sorted_dispatch: ids must be [.., T, k={}], got {}",
        top_k,
        ids.logical_shape());
    Op::operation_attributes_t attrs{
        .num_experts = num_experts,
        .top_k = top_k,
        .group = group,
        .num_tokens = R / top_k,
        .dispatched_rows = dispatched_rows,
        .num_groups = dispatched_rows / group,
        .pad_group_id = pad_group_id,
        .output_mem_config = memory_config.value_or(ids.memory_config()),
    };
    return ttnn::device_operation::launch<Op>(attrs, Op::tensor_args_t{ids, scores});
}

}  // namespace ttnn::prim
