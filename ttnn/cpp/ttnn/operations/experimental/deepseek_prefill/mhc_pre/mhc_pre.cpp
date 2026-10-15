// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_pre.hpp"

#include "device/mhc_pre_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::mhc_pre {

ttnn::Tensor mhc_pre(
    const ttnn::Tensor& residual,
    const ttnn::Tensor& pre,
    const ttnn::Tensor& consts,
    uint32_t n,
    std::optional<tt::tt_metal::DataType> output_dtype) {
    return ttnn::prim::mhc_pre(residual, pre, consts, n, output_dtype);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::mhc_pre
