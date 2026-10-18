// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "kda_decode_conv_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::experimental::prim {

namespace {
void check(const Tensor& t, const char* name, tt::tt_metal::DataType dtype) {
    TT_FATAL(
        t.storage_type() == StorageType::DEVICE && t.is_allocated(), "kda_decode_conv: {} must be on device", name);
    TT_FATAL(t.dtype() == dtype, "kda_decode_conv: {} must be {}, got {}", name, dtype, t.dtype());
    TT_FATAL(t.layout() == tt::tt_metal::Layout::TILE, "kda_decode_conv: {} must be TILE layout", name);
    TT_FATAL(!t.memory_config().is_sharded(), "kda_decode_conv: {} must be interleaved", name);
    TT_FATAL(t.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM, "kda_decode_conv: {} must be in DRAM", name);
    const auto& ps = t.padded_shape();
    TT_FATAL(ps.rank() == 4 && ps[0] == 1 && ps[1] == 1, "kda_decode_conv: {} must be [1, 1, R, C]", name);
    TT_FATAL(ps[2] == tt::constants::TILE_HEIGHT, "kda_decode_conv: {} must be one tile row (rows <= 32)", name);
}
}  // namespace

void KdaDecodeConvDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& t) {
    using tt::tt_metal::DataType;
    const uint32_t W = args.width;
    TT_FATAL(W > 0 && W % tt::constants::TILE_WIDTH == 0, "kda_decode_conv: width {} must be a multiple of 32", W);
    check(t.proj, "proj", DataType::BFLOAT16);
    check(t.h0, "h0", DataType::BFLOAT16);
    check(t.h1, "h1", DataType::BFLOAT16);
    check(t.h2, "h2", DataType::BFLOAT16);
    check(t.tap0, "tap0", DataType::FLOAT32);
    check(t.tap1, "tap1", DataType::FLOAT32);
    check(t.tap2, "tap2", DataType::FLOAT32);
    check(t.tap3, "tap3", DataType::FLOAT32);
    check(t.active, "active", DataType::FLOAT32);
    TT_FATAL(t.proj.padded_shape()[3] >= W, "kda_decode_conv: proj must have >= width columns");
    for (const Tensor* h : {&t.h0, &t.h1, &t.h2, &t.tap0, &t.tap1, &t.tap2, &t.tap3}) {
        TT_FATAL(h->padded_shape()[3] == W, "kda_decode_conv: history / tap width must equal width {}", W);
    }
    const uint32_t B = t.proj.logical_shape()[2];
    for (const Tensor* h : {&t.h0, &t.h1, &t.h2, &t.active}) {
        TT_FATAL(h->logical_shape()[2] == B, "kda_decode_conv: history / active rows must equal proj rows {}", B);
    }
    auto* dev = t.proj.device();
    for (const Tensor* h : {&t.h0, &t.h1, &t.h2, &t.tap0, &t.tap1, &t.tap2, &t.tap3, &t.active}) {
        TT_FATAL(h->device() == dev, "kda_decode_conv: all inputs must be on the same device");
    }
}

KdaDecodeConvDeviceOperation::spec_return_value_t KdaDecodeConvDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& t) {
    const uint32_t B = t.proj.logical_shape()[2];
    return tt::tt_metal::TensorSpec(
        ttnn::Shape({1, 1, B, args.width}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::FLOAT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM}));
}

KdaDecodeConvDeviceOperation::tensor_return_value_t KdaDecodeConvDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& t) {
    return create_device_tensor(compute_output_specs(args, t), t.proj.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor kda_decode_conv(
    const Tensor& proj,
    const Tensor& h0,
    const Tensor& h1,
    const Tensor& h2,
    const Tensor& tap0,
    const Tensor& tap1,
    const Tensor& tap2,
    const Tensor& tap3,
    const Tensor& active,
    uint32_t width) {
    using OperationType = ttnn::experimental::prim::KdaDecodeConvDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{width};
    auto tensor_args = OperationType::tensor_args_t{proj, h0, h1, h2, tap0, tap1, tap2, tap3, active};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
