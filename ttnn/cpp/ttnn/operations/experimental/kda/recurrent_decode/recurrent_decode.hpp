// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::kda {

// One KDA (Kimi Delta Attention) decode step for every (user, head), state updated in place:
//
//     qn = l2norm(q) * q_scale,  kn = l2norm(k)              (x / sqrt(sum x^2 + eps), per head)
//     Sd = diag(decay) S                                     (per key-channel decay, decay = exp(g))
//     u  = beta * (v - kn Sd)
//     S <- Sd + kn^T u
//     o  = qn S
//
//   state: [B, H, K, V] FLOAT32 TILE, updated in place (fixed address: trace-safe)
//   qkv:   [1, 1, R, 2*H*K + H*V] FLOAT32 TILE, R >= B: user b in row b, q | k | v head-major column blocks
//   decay: [1, 1, R, H*K] FLOAT32 TILE (exp already applied)
//   beta:  [1, 1, R, H]   FLOAT32 TILE (sigmoid already applied)
//
// Returns o [1, 1, B*H, V] FLOAT32 TILE (row b*H + h).
ttnn::Tensor recurrent_decode(
    const ttnn::Tensor& state,
    const ttnn::Tensor& qkv,
    const ttnn::Tensor& decay,
    const ttnn::Tensor& beta,
    float q_scale,
    float eps = 1e-6f);

}  // namespace ttnn::experimental::kda
