// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "scatter_rows_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

void ScatterRowsDeviceOperation::validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t& t) {
    TT_FATAL(
        t.cache.storage_type() == StorageType::DEVICE && t.src.storage_type() == StorageType::DEVICE &&
            t.row_idx.storage_type() == StorageType::DEVICE,
        "scatter_rows: operands must be on device");
    TT_FATAL(
        t.cache.layout() == Layout::ROW_MAJOR && t.src.layout() == Layout::ROW_MAJOR &&
            t.row_idx.layout() == Layout::ROW_MAJOR,
        "scatter_rows: operands must be ROW_MAJOR");
    TT_FATAL(
        !t.cache.is_sharded() && !t.src.is_sharded() && !t.row_idx.is_sharded(),
        "scatter_rows: interleaved operands only");
    TT_FATAL(t.cache.dtype() == t.src.dtype(), "scatter_rows: cache and src dtypes differ");
    TT_FATAL(
        t.cache.logical_shape()[-1] == t.src.logical_shape()[-1],
        "scatter_rows: row widths differ ({} vs {})",
        t.cache.logical_shape()[-1],
        t.src.logical_shape()[-1]);
    TT_FATAL(
        t.row_idx.dtype() == DataType::UINT32 || t.row_idx.dtype() == DataType::INT32,
        "scatter_rows: row_idx must be uint32 or int32");
    const uint32_t W = t.src.logical_shape()[-1];
    const uint32_t R = t.src.logical_volume() / W;
    TT_FATAL(
        t.row_idx.logical_volume() == R && t.row_idx.logical_shape()[-1] == R,
        "scatter_rows: row_idx must be one stick of R = {} indices, got {}",
        R,
        t.row_idx.logical_shape());
}

void ScatterRowsDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& a, const tensor_args_t& t) {
    validate_on_program_cache_miss(a, t);
}

ScatterRowsDeviceOperation::spec_return_value_t ScatterRowsDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& t) {
    return t.cache.tensor_spec();
}

ScatterRowsDeviceOperation::tensor_return_value_t ScatterRowsDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& t) {
    return t.cache;  // in place
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor scatter_rows(const Tensor& cache, const Tensor& src, const Tensor& row_idx) {
    using Op = ttnn::experimental::prim::ScatterRowsDeviceOperation;
    return ttnn::device_operation::launch<Op>(Op::operation_attributes_t{}, Op::tensor_args_t{cache, src, row_idx});
}

}  // namespace ttnn::prim
