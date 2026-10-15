// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "recurrent_decode.hpp"

#include "device/recurrent_decode_device_operation.hpp"

namespace ttnn::experimental::kda {

ttnn::Tensor recurrent_decode(
    const ttnn::Tensor& state,
    const ttnn::Tensor& qkv,
    const ttnn::Tensor& decay,
    const ttnn::Tensor& beta,
    float q_scale,
    float eps) {
    return ttnn::prim::kda_recurrent_decode(state, qkv, decay, beta, q_scale, eps);
}

}  // namespace ttnn::experimental::kda
