// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_gather_add_rows_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

void MoeGatherAddRowsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& a, const tensor_args_t& t) {
    const uint32_t R = a.num_tokens * a.top_k;
    TT_FATAL(
        t.rows.storage_type() == StorageType::DEVICE && t.undispatch.storage_type() == StorageType::DEVICE,
        "moe_gather_add_rows: operands must be on device");
    TT_FATAL(
        t.rows.dtype() == DataType::BFLOAT16, "moe_gather_add_rows: rows must be bfloat16, got {}", t.rows.dtype());
    TT_FATAL(
        t.undispatch.dtype() == DataType::UINT32,
        "moe_gather_add_rows: undispatch must be uint32, got {}",
        t.undispatch.dtype());
    TT_FATAL(
        t.rows.layout() == Layout::ROW_MAJOR && t.undispatch.layout() == Layout::ROW_MAJOR,
        "moe_gather_add_rows: rows / undispatch must be ROW_MAJOR");
    TT_FATAL(!t.rows.is_sharded() && !t.undispatch.is_sharded(), "moe_gather_add_rows: interleaved operands only");
    TT_FATAL(
        t.rows.logical_shape().rank() == 2,
        "moe_gather_add_rows: rows must be [Rd, H], got {}",
        t.rows.logical_shape());
    TT_FATAL(t.rows.logical_shape()[-1] % 32 == 0, "moe_gather_add_rows: H must be a multiple of 32");
    TT_FATAL(
        t.undispatch.logical_volume() == R && t.undispatch.logical_shape()[-1] == R,
        "moe_gather_add_rows: undispatch must be one stick [1, T*k = {}], got {}",
        R,
        t.undispatch.logical_shape());
    TT_FATAL(a.top_k >= 1 && a.top_k <= 32, "moe_gather_add_rows: top_k must be in [1, 32]");
}

void MoeGatherAddRowsDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& a, const tensor_args_t& t) {
    validate_on_program_cache_miss(a, t);
}

MoeGatherAddRowsDeviceOperation::spec_return_value_t MoeGatherAddRowsDeviceOperation::compute_output_specs(
    const operation_attributes_t& a, const tensor_args_t& t) {
    const uint32_t H = t.rows.logical_shape()[-1];
    return TensorSpec(
        ttnn::Shape({1, 1, a.num_tokens, H}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), a.output_mem_config));
}

MoeGatherAddRowsDeviceOperation::tensor_return_value_t MoeGatherAddRowsDeviceOperation::create_output_tensors(
    const operation_attributes_t& a, const tensor_args_t& t) {
    return create_device_tensor(compute_output_specs(a, t), t.rows.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor moe_gather_add_rows(
    const Tensor& rows,
    const Tensor& undispatch,
    uint32_t num_tokens,
    uint32_t top_k,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config) {
    using Op = ttnn::experimental::prim::MoeGatherAddRowsDeviceOperation;
    Op::operation_attributes_t attrs{
        .num_tokens = num_tokens,
        .top_k = top_k,
        .output_mem_config = memory_config.value_or(rows.memory_config()),
    };
    return ttnn::device_operation::launch<Op>(attrs, Op::tensor_args_t{rows, undispatch});
}

}  // namespace ttnn::prim
