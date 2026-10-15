// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct MhcPreParams {
    uint32_t n;                           // expansion rate (streams)
    tt::tt_metal::DataType output_dtype;  // dtype of h
};

struct MhcPreTensorArgs {
    const Tensor& residual;
    const Tensor& pre;
    const Tensor& consts;
};

}  // namespace ttnn::experimental::prim
