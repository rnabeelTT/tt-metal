// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_pre_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::experimental::prim {

namespace {

// Inputs are indexed by tile page: an unpadded rank-4 [1,1,rows,cols] block with tile-aligned dims keeps host and
// kernel page arithmetic the same expression (see mhc_post).
void check_flat_2d(const Tensor& t, const char* name, uint32_t rows, uint32_t cols, tt::tt_metal::DataType dtype) {
    TT_FATAL(t.dtype() == dtype, "{} must have the residual's dtype ({}), got {}", name, dtype, t.dtype());
    TT_FATAL(t.layout() == tt::tt_metal::Layout::TILE, "{} must be TILE layout", name);
    TT_FATAL(t.is_allocated(), "{} must be allocated on device", name);
    TT_FATAL(t.storage_type() == StorageType::DEVICE, "{} must be a device tensor", name);
    const auto& ls = t.logical_shape();
    TT_FATAL(ls.rank() >= 2, "{} must be rank>=2, got rank {}", name, ls.rank());
    for (int i = 0; i + 2 < static_cast<int>(ls.rank()); ++i) {
        TT_FATAL(ls[i] == 1, "{} leading dim {} must be 1 (got {})", name, i, ls[i]);
    }
    TT_FATAL(
        static_cast<uint32_t>(ls[-2]) == rows && static_cast<uint32_t>(ls[-1]) == cols,
        "{} must be [{}, {}], got [{}, {}]",
        name,
        rows,
        cols,
        ls[-2],
        ls[-1]);
}

}  // namespace

void MhcPreDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const uint32_t n = args.n;
    TT_FATAL(n > 0 && n <= tt::constants::TILE_WIDTH, "n must be in (0, 32]");
    const auto& rls = tensor_args.residual.logical_shape();
    TT_FATAL(rls.rank() >= 2, "residual must be rank>=2");
    const uint32_t T = rls[-2];
    const uint32_t nC = rls[-1];
    TT_FATAL(nC % n == 0, "residual width {} must be n * C (n = {})", nC, n);
    const uint32_t C = nC / n;
    TT_FATAL(C % tt::constants::TILE_WIDTH == 0, "stream width ({}) must be a multiple of 32", C);
    const auto dtype = tensor_args.residual.dtype();
    TT_FATAL(
        dtype == tt::tt_metal::DataType::FLOAT32 || dtype == tt::tt_metal::DataType::BFLOAT16,
        "residual must be FLOAT32 or BFLOAT16, got {}",
        dtype);
    TT_FATAL(
        args.output_dtype == tt::tt_metal::DataType::FLOAT32 || args.output_dtype == tt::tt_metal::DataType::BFLOAT16,
        "output_dtype must be FLOAT32 or BFLOAT16");
    check_flat_2d(tensor_args.residual, "residual", T, nC, dtype);
    check_flat_2d(tensor_args.pre, "pre", T, n, dtype);
    const auto& consts = tensor_args.consts;
    TT_FATAL(consts.dtype() == dtype, "consts must have the residual's dtype");
    TT_FATAL(consts.layout() == tt::tt_metal::Layout::TILE, "consts must be TILE layout");
    TT_FATAL(consts.is_allocated() && consts.storage_type() == StorageType::DEVICE, "consts must be on device");
    TT_FATAL(
        consts.physical_volume() / tt::constants::TILE_HW >= n,
        "consts must hold at least n column-broadcast tiles, got {}",
        consts.physical_volume() / tt::constants::TILE_HW);
    auto* dev = tensor_args.residual.device();
    TT_FATAL(tensor_args.pre.device() == dev && consts.device() == dev, "all inputs must be on the same device");
}

MhcPreDeviceOperation::spec_return_value_t MhcPreDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& r = tensor_args.residual;
    return tt::tt_metal::TensorSpec(
        ttnn::Shape({1, 1, r.logical_shape()[-2], r.logical_shape()[-1] / args.n}),
        tt::tt_metal::TensorLayout(
            args.output_dtype,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM}));
}

MhcPreDeviceOperation::tensor_return_value_t MhcPreDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.residual.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor mhc_pre(
    const Tensor& residual,
    const Tensor& pre,
    const Tensor& consts,
    uint32_t n,
    std::optional<tt::tt_metal::DataType> output_dtype) {
    using OperationType = ttnn::experimental::prim::MhcPreDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{n, output_dtype.value_or(residual.dtype())};
    auto tensor_args = OperationType::tensor_args_t{residual, pre, consts};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
