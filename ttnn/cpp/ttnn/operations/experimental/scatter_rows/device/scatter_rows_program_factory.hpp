// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "scatter_rows_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct ScatterRowsProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle kernel{};
        std::vector<CoreCoord> cores;
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const ScatterRowsParams&, const ScatterRowsInputs& tensor_args, Tensor& out);

    static void override_runtime_arguments(
        cached_program_t& cached_program, const ScatterRowsParams&, const ScatterRowsInputs& tensor_args, Tensor& out);
};

}  // namespace ttnn::experimental::prim
