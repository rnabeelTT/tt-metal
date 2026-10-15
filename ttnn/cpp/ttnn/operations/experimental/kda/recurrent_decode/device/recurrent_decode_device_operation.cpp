// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "recurrent_decode_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::experimental::prim {

namespace {

using tt::tt_metal::DataType;
using tt::tt_metal::Layout;

void check_input(const Tensor& t, const char* name) {
    TT_FATAL(
        t.storage_type() == StorageType::DEVICE && t.is_allocated(),
        "kda.recurrent_decode: {} must be on device",
        name);
    TT_FATAL(t.dtype() == DataType::FLOAT32, "kda.recurrent_decode: {} must be FLOAT32, got {}", name, t.dtype());
    TT_FATAL(t.layout() == Layout::TILE, "kda.recurrent_decode: {} must be TILE layout", name);
    TT_FATAL(!t.is_sharded(), "kda.recurrent_decode: {} must be interleaved", name);
    TT_FATAL(
        t.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "kda.recurrent_decode: {} must be in DRAM (the reader addresses 64-byte face rows)",
        name);
}

// [1, 1, rows >= B, cols] with the given width.
void check_rows(const Tensor& t, const char* name, uint32_t batch, uint32_t cols) {
    const auto& s = t.logical_shape();
    TT_FATAL(s.rank() == 4 && s[0] == 1 && s[1] == 1, "kda.recurrent_decode: {} must be [1, 1, R, C]", name);
    TT_FATAL(s[2] >= batch, "kda.recurrent_decode: {} has {} rows < batch {}", name, s[2], batch);
    TT_FATAL(s[3] == cols, "kda.recurrent_decode: {} must have {} columns, got {}", name, cols, s[3]);
}

}  // namespace

void KdaRecurrentDecodeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& a, const tensor_args_t& t) {
    check_input(t.state, "state");
    check_input(t.qkv, "qkv");
    check_input(t.decay, "decay");
    check_input(t.beta, "beta");
    TT_FATAL(
        a.key_dim % tt::constants::TILE_WIDTH == 0 && a.value_dim % tt::constants::TILE_WIDTH == 0,
        "kda.recurrent_decode: K and V must be multiples of 32");
    TT_FATAL(
        (a.batch * a.num_heads) % tt::constants::TILE_HEIGHT == 0,
        "kda.recurrent_decode: B * H must be a multiple of 32 (output rows)");
    const auto& ss = t.state.logical_shape();
    TT_FATAL(ss.rank() == 4, "kda.recurrent_decode: state must be [B, H, K, V]");
    check_rows(t.qkv, "qkv", a.batch, a.num_heads * (2 * a.key_dim + a.value_dim));
    check_rows(t.decay, "decay", a.batch, a.num_heads * a.key_dim);
    check_rows(t.beta, "beta", a.batch, a.num_heads);
    auto* dev = t.state.device();
    TT_FATAL(
        t.qkv.device() == dev && t.decay.device() == dev && t.beta.device() == dev,
        "kda.recurrent_decode: all inputs must be on the same device");
}

KdaRecurrentDecodeDeviceOperation::spec_return_value_t KdaRecurrentDecodeDeviceOperation::compute_output_specs(
    const operation_attributes_t& a, const tensor_args_t&) {
    return tt::tt_metal::TensorSpec(
        ttnn::Shape({1, 1, a.batch * a.num_heads, a.value_dim}),
        tt::tt_metal::TensorLayout(
            DataType::FLOAT32,
            tt::tt_metal::PageConfig(Layout::TILE),
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM}));
}

KdaRecurrentDecodeDeviceOperation::tensor_return_value_t KdaRecurrentDecodeDeviceOperation::create_output_tensors(
    const operation_attributes_t& a, const tensor_args_t& t) {
    return create_device_tensor(compute_output_specs(a, t), t.state.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor kda_recurrent_decode(
    const Tensor& state, const Tensor& qkv, const Tensor& decay, const Tensor& beta, float q_scale, float eps) {
    using OperationType = ttnn::experimental::prim::KdaRecurrentDecodeDeviceOperation;
    const auto& s = state.logical_shape();
    TT_FATAL(s.rank() == 4, "kda.recurrent_decode: state must be [B, H, K, V]");
    auto attrs = OperationType::operation_attributes_t{
        static_cast<uint32_t>(s[0]),
        static_cast<uint32_t>(s[1]),
        static_cast<uint32_t>(s[2]),
        static_cast<uint32_t>(s[3]),
        q_scale,
        eps};
    auto args = OperationType::tensor_args_t{state, qkv, decay, beta};
    return ttnn::device_operation::launch<OperationType>(attrs, args);
}

}  // namespace ttnn::prim
