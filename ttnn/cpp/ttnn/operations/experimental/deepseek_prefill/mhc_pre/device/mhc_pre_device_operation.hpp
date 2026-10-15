// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "mhc_pre_program_factory.hpp"
#include "mhc_pre_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct MhcPreDeviceOperation {
    using operation_attributes_t = MhcPreParams;
    using tensor_args_t = MhcPreTensorArgs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<MhcPreProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
Tensor mhc_pre(
    const Tensor& residual,
    const Tensor& pre,
    const Tensor& consts,
    uint32_t n,
    std::optional<tt::tt_metal::DataType> output_dtype = std::nullopt);
}  // namespace ttnn::prim
