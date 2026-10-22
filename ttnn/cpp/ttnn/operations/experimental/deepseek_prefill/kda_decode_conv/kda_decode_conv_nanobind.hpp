// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace ttnn::operations::experimental::deepseek_prefill::kda_decode_conv::detail {
void bind_experimental_kda_decode_conv_operation(nb::module_& mod);
}  // namespace ttnn::operations::experimental::deepseek_prefill::kda_decode_conv::detail
