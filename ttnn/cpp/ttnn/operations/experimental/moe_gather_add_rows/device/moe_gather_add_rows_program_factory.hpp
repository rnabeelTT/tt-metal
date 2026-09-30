// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "moe_gather_add_rows_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct MoeGatherAddRowsProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader{};
        tt::tt_metal::KernelHandle writer{};
        std::vector<CoreCoord> cores;
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const MoeGatherAddRowsParams& operation_attributes,
        const MoeGatherAddRowsInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const MoeGatherAddRowsParams& operation_attributes,
        const MoeGatherAddRowsInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
