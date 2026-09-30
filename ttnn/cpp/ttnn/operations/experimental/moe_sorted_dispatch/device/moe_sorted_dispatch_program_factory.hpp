// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "moe_sorted_dispatch_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct MoeSortedDispatchProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle kernel{};
        CoreCoord core;
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const MoeSortedDispatchParams& operation_attributes,
        const MoeSortedDispatchInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const MoeSortedDispatchParams& operation_attributes,
        const MoeSortedDispatchInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
