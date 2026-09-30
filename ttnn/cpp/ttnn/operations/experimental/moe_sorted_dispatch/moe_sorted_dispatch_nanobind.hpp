// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::moe_dispatch_nb::detail {
void bind_experimental_moe_sorted_dispatch_operation(nb::module_& mod);
}  // namespace ttnn::operations::experimental::moe_dispatch_nb::detail
