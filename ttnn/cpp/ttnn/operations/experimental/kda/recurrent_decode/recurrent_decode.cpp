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
    float eps,
    uint32_t num_qk_heads,
    bool per_head_decay,
    uint32_t batch) {
    return ttnn::prim::kda_recurrent_decode(state, qkv, decay, beta, q_scale, eps, num_qk_heads, per_head_decay, batch);
}

}  // namespace ttnn::experimental::kda
