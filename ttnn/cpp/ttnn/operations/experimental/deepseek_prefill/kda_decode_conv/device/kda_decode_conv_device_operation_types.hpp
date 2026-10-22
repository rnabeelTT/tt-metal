// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct KdaDecodeConvParams {
    uint32_t width;  // conv width W (columns of proj / the history / the taps)
};

struct KdaDecodeConvTensorArgs {
    const Tensor& proj;
    const Tensor& h0;
    const Tensor& h1;
    const Tensor& h2;
    const Tensor& tap0;
    const Tensor& tap1;
    const Tensor& tap2;
    const Tensor& tap3;
    const Tensor& active;
};

}  // namespace ttnn::experimental::prim
