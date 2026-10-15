// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>
namespace nb = nanobind;

namespace ttnn::operations::experimental::scatter_rows_nb::detail {
void bind_experimental_scatter_rows_operation(nb::module_& mod);
}  // namespace ttnn::operations::experimental::scatter_rows_nb::detail
