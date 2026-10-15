// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "dsa_decode_scores_program_factory.hpp"
#include "dsa_decode_scores_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct DsaDecodeScoresDeviceOperation {
    using operation_attributes_t = DsaDecodeScoresParams;
    using tensor_args_t = DsaDecodeScoresTensorArgs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<DsaDecodeScoresProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
Tensor dsa_decode_scores(
    const Tensor& q,
    const Tensor& weights,
    const Tensor& k_pool,
    const Tensor& page_table,
    const Tensor& limits,
    uint32_t num_blocks);
}  // namespace ttnn::prim
