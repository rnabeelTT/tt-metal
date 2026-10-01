// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "moe_gather_rows_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

void MoeGatherRowsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& a, const tensor_args_t& t) {
    TT_FATAL(
        t.rows.storage_type() == StorageType::DEVICE && t.src_tok.storage_type() == StorageType::DEVICE &&
            t.group_ids.storage_type() == StorageType::DEVICE,
        "moe_gather_rows: operands must be on device");
    TT_FATAL(t.rows.dtype() == DataType::BFLOAT16, "moe_gather_rows: rows must be bfloat16, got {}", t.rows.dtype());
    TT_FATAL(
        t.src_tok.dtype() == DataType::UINT32, "moe_gather_rows: src_tok must be uint32, got {}", t.src_tok.dtype());
    TT_FATAL(
        t.group_ids.dtype() == DataType::UINT16,
        "moe_gather_rows: group_ids must be uint16, got {}",
        t.group_ids.dtype());
    TT_FATAL(
        t.rows.layout() == Layout::ROW_MAJOR && t.src_tok.layout() == Layout::ROW_MAJOR &&
            t.group_ids.layout() == Layout::ROW_MAJOR,
        "moe_gather_rows: operands must be ROW_MAJOR");
    TT_FATAL(
        !t.rows.is_sharded() && !t.src_tok.is_sharded() && !t.group_ids.is_sharded(),
        "moe_gather_rows: interleaved operands only");
    TT_FATAL(
        t.rows.logical_shape().rank() == 2, "moe_gather_rows: rows must be [T, H], got {}", t.rows.logical_shape());
    TT_FATAL(t.rows.logical_shape()[-1] % 32 == 0, "moe_gather_rows: H must be a multiple of 32");
    const uint32_t G = t.group_ids.logical_shape()[-1];
    TT_FATAL(
        t.group_ids.logical_volume() == G && G >= 1,
        "moe_gather_rows: group_ids must be one stick [1, G], got {}",
        t.group_ids.logical_shape());
    TT_FATAL(
        t.src_tok.logical_volume() == G * 32 && t.src_tok.logical_shape()[-1] == G * 32,
        "moe_gather_rows: src_tok must be one stick [1, G*32 = {}], got {}",
        G * 32,
        t.src_tok.logical_shape());
    TT_FATAL(a.num_experts >= 1, "moe_gather_rows: num_experts must be >= 1");
}

void MoeGatherRowsDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& a, const tensor_args_t& t) {
    validate_on_program_cache_miss(a, t);
}

MoeGatherRowsDeviceOperation::spec_return_value_t MoeGatherRowsDeviceOperation::compute_output_specs(
    const operation_attributes_t& a, const tensor_args_t& t) {
    const uint32_t H = t.rows.logical_shape()[-1];
    const uint32_t G = t.group_ids.logical_shape()[-1];
    return TensorSpec(
        ttnn::Shape({1, G, 32, H}), TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), a.output_mem_config));
}

MoeGatherRowsDeviceOperation::tensor_return_value_t MoeGatherRowsDeviceOperation::create_output_tensors(
    const operation_attributes_t& a, const tensor_args_t& t) {
    return create_device_tensor(compute_output_specs(a, t), t.rows.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor moe_gather_rows(
    const Tensor& rows,
    const Tensor& src_tok,
    const Tensor& group_ids,
    uint32_t num_experts,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config) {
    using Op = ttnn::experimental::prim::MoeGatherRowsDeviceOperation;
    Op::operation_attributes_t attrs{
        .num_experts = num_experts,
        .output_mem_config = memory_config.value_or(rows.memory_config()),
    };
    return ttnn::device_operation::launch<Op>(attrs, Op::tensor_args_t{rows, src_tok, group_ids});
}

}  // namespace ttnn::prim
