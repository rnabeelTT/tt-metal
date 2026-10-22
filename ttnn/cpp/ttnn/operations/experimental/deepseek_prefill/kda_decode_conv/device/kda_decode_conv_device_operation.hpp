// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "kda_decode_conv_program_factory.hpp"
#include "kda_decode_conv_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct KdaDecodeConvDeviceOperation {
    using operation_attributes_t = KdaDecodeConvParams;
    using tensor_args_t = KdaDecodeConvTensorArgs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<KdaDecodeConvProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

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
    uint32_t width);
}  // namespace ttnn::prim
