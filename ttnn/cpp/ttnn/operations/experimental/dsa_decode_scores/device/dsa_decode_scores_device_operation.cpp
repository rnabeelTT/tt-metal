// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dsa_decode_scores_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::experimental::prim {

namespace {

using tt::tt_metal::BufferType;
using tt::tt_metal::DataType;
using tt::tt_metal::Layout;
constexpr uint32_t HEADS = 32;

void check_dram(const Tensor& t, const char* name, Layout layout) {
    TT_FATAL(
        t.storage_type() == StorageType::DEVICE && t.is_allocated(), "dsa_decode_scores: {} must be on device", name);
    TT_FATAL(t.layout() == layout, "dsa_decode_scores: {} has the wrong layout", name);
    TT_FATAL(
        !t.is_sharded() && t.buffer()->buffer_type() == BufferType::DRAM,
        "dsa_decode_scores: {} must be DRAM interleaved",
        name);
}

}  // namespace

void DsaDecodeScoresDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& a, const tensor_args_t& t) {
    check_dram(t.q, "q", Layout::TILE);
    check_dram(t.weights, "weights", Layout::TILE);
    check_dram(t.k_pool, "k_pool", Layout::TILE);
    check_dram(t.page_table, "page_table", Layout::ROW_MAJOR);
    check_dram(t.limits, "limits", Layout::ROW_MAJOR);
    TT_FATAL(t.q.dtype() == DataType::BFLOAT16 && t.k_pool.dtype() == DataType::BFLOAT16, "q, k_pool must be bf16");
    TT_FATAL(t.weights.dtype() == DataType::FLOAT32, "weights must be fp32");
    TT_FATAL(
        t.page_table.dtype() == DataType::INT32 || t.page_table.dtype() == DataType::UINT32,
        "page_table must be int32");
    TT_FATAL(t.limits.dtype() == DataType::UINT32, "limits must be uint32");
    const auto& ws = t.weights.logical_shape();
    const uint32_t S = ws[-2];
    TT_FATAL(ws[-1] == HEADS, "weights must be [1, 1, S, 32]");
    const auto& qs = t.q.logical_shape();
    const uint32_t D = qs[-1];
    TT_FATAL(qs[-2] == S * HEADS, "q must be [1, 1, S*32, D] (32 heads per slot)");
    TT_FATAL(D % tt::constants::TILE_WIDTH == 0 && t.k_pool.logical_shape()[-1] == D, "k_pool last dim must be D");
    TT_FATAL(t.page_table.logical_shape()[0] >= S, "page_table needs one row per slot");
    TT_FATAL(t.page_table.logical_shape()[-1] >= a.num_blocks, "page_table rows must cover num_blocks");
    TT_FATAL(t.limits.logical_volume() >= S, "limits must hold one value per slot");
    TT_FATAL(a.num_blocks > 0, "num_blocks must be positive");
}

DsaDecodeScoresDeviceOperation::spec_return_value_t DsaDecodeScoresDeviceOperation::compute_output_specs(
    const operation_attributes_t& a, const tensor_args_t& t) {
    return tt::tt_metal::TensorSpec(
        ttnn::Shape({1, 1, t.weights.logical_shape()[-2], a.num_blocks * tt::constants::TILE_WIDTH}),
        tt::tt_metal::TensorLayout(
            DataType::FLOAT32,
            tt::tt_metal::PageConfig(Layout::TILE),
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}));
}

DsaDecodeScoresDeviceOperation::tensor_return_value_t DsaDecodeScoresDeviceOperation::create_output_tensors(
    const operation_attributes_t& a, const tensor_args_t& t) {
    return create_device_tensor(compute_output_specs(a, t), t.q.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor dsa_decode_scores(
    const Tensor& q,
    const Tensor& weights,
    const Tensor& k_pool,
    const Tensor& page_table,
    const Tensor& limits,
    uint32_t num_blocks) {
    using OperationType = ttnn::experimental::prim::DsaDecodeScoresDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{num_blocks},
        OperationType::tensor_args_t{q, weights, k_pool, page_table, limits});
}

}  // namespace ttnn::prim
