// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_mix_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::experimental::prim {

using tt::constants::TILE_HEIGHT;
using tt::constants::TILE_WIDTH;

void MhcMixDeviceOperation::validate_on_program_cache_miss(const operation_attributes_t& args, const tensor_args_t& t) {
    for (const Tensor* p : {&t.x, &t.fn_t}) {
        TT_FATAL(p->storage_type() == StorageType::DEVICE && p->is_allocated(), "mhc_mix: inputs must be on device");
        TT_FATAL(p->dtype() == tt::tt_metal::DataType::FLOAT32, "mhc_mix: inputs must be FLOAT32");
        TT_FATAL(p->layout() == tt::tt_metal::Layout::TILE, "mhc_mix: inputs must be TILE");
        TT_FATAL(!p->is_sharded(), "mhc_mix: inputs must be interleaved");
    }
    const auto& xs = t.x.logical_shape();
    TT_FATAL(xs.rank() == 4 && xs[0] == 1 && xs[1] == 1, "mhc_mix: x must be [1, 1, T, K]");
    const uint32_t K = xs[3];
    TT_FATAL(K % TILE_WIDTH == 0, "mhc_mix: K must be a tile multiple");  // T: any (the padded tile rows are processed)
    const auto& fs = t.fn_t.logical_shape();
    TT_FATAL(fs[-2] == K && fs[-1] == TILE_WIDTH, "mhc_mix: fn_t must be [K, 32]");
    TT_FATAL(
        args.num_chunks > 0 && (K / TILE_WIDTH) % args.num_chunks == 0,
        "mhc_mix: num_chunks must divide the {} K tiles",
        K / TILE_WIDTH);
    TT_FATAL(t.x.device() == t.fn_t.device(), "mhc_mix: inputs must be on one device");
}

MhcMixDeviceOperation::spec_return_value_t MhcMixDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& t) {
    return tt::tt_metal::TensorSpec(
        ttnn::Shape({1, args.num_chunks, t.x.logical_shape()[2], 2 * TILE_WIDTH}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::FLOAT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM}));
}

MhcMixDeviceOperation::tensor_return_value_t MhcMixDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& t) {
    return create_device_tensor(compute_output_specs(args, t), t.x.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor mhc_mix(const Tensor& x, const Tensor& fn_t, uint32_t num_chunks) {
    using OperationType = ttnn::experimental::prim::MhcMixDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{num_chunks}, OperationType::tensor_args_t{x, fn_t});
}

}  // namespace ttnn::prim
