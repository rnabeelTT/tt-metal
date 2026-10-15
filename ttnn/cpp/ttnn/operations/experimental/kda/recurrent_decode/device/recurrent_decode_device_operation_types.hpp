// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct KdaRecurrentDecodeParams {
    uint32_t batch;      // B (users = state.shape[0])
    uint32_t num_heads;  // H
    uint32_t key_dim;    // K
    uint32_t value_dim;  // V
    float q_scale;
    float eps;
};

struct KdaRecurrentDecodeTensorArgs {
    const Tensor& state;
    const Tensor& qkv;
    const Tensor& decay;
    const Tensor& beta;
};

}  // namespace ttnn::experimental::prim
