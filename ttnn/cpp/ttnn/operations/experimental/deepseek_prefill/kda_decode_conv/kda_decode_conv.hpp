// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::kda_decode_conv {

// One KDA decode step of the short causal conv (4 taps) + SiLU for B <= 32 slots, fused: see kda_decode_conv_nanobind.
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
    uint32_t width);

}  // namespace ttnn::operations::experimental::deepseek_prefill::kda_decode_conv

namespace ttnn {
using operations::experimental::deepseek_prefill::kda_decode_conv::kda_decode_conv;
}  // namespace ttnn
