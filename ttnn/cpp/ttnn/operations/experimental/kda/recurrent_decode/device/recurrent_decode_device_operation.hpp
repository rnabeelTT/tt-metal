// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "recurrent_decode_program_factory.hpp"
#include "recurrent_decode_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct KdaRecurrentDecodeDeviceOperation {
    using operation_attributes_t = KdaRecurrentDecodeParams;
    using tensor_args_t = KdaRecurrentDecodeTensorArgs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<KdaRecurrentDecodeProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
Tensor kda_recurrent_decode(
    const Tensor& state, const Tensor& qkv, const Tensor& decay, const Tensor& beta, float q_scale, float eps);
}  // namespace ttnn::prim
