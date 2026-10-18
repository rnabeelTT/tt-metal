// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "kda_decode_conv.hpp"
#include "device/kda_decode_conv_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::kda_decode_conv {

ttnn::Tensor kda_decode_conv(
    const ttnn::Tensor& proj,
    const ttnn::Tensor& h0,
    const ttnn::Tensor& h1,
    const ttnn::Tensor& h2,
    const ttnn::Tensor& tap0,
    const ttnn::Tensor& tap1,
    const ttnn::Tensor& tap2,
    const ttnn::Tensor& tap3,
    const ttnn::Tensor& active,
    uint32_t width) {
    return ttnn::prim::kda_decode_conv(proj, h0, h1, h2, tap0, tap1, tap2, tap3, active, width);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::kda_decode_conv
