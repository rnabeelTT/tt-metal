// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "mhc_mix_program_factory.hpp"
#include "mhc_mix_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct MhcMixDeviceOperation {
    using operation_attributes_t = MhcMixParams;
    using tensor_args_t = MhcMixTensorArgs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<MhcMixProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
Tensor mhc_mix(const Tensor& x, const Tensor& fn_t, uint32_t num_chunks);
}  // namespace ttnn::prim
