// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>
namespace nb = nanobind;

namespace ttnn::operations::experimental::dsa_decode_scores_nb::detail {
void bind_experimental_dsa_decode_scores_operation(nb::module_& mod);
}  // namespace ttnn::operations::experimental::dsa_decode_scores_nb::detail
