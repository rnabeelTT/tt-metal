// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "recurrent_decode_nanobind.hpp"
#include "recurrent_decode.hpp"

#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::kda::recurrent_decode::detail {

void bind_recurrent_decode(nb::module_& mod) {
    ttnn::bind_function<"recurrent_decode", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        One KDA (Kimi Delta Attention) decode step for every (user, head); the recurrent state is updated in place.

        Per head (b, h), all in fp32:

            qn = q / sqrt(sum(q^2) + eps) * q_scale,   kn = k / sqrt(sum(k^2) + eps)
            Sd = diag(decay) S
            u  = beta * (v - kn Sd)
            S  = Sd + kn^T u
            o  = qn S

        The per-head rows are read directly from the packed projection layouts (no head split / relayout).

        Args:
            state (ttnn.Tensor): ``[B, H, K, V]`` FLOAT32 TILE recurrent state, updated in place.
            qkv (ttnn.Tensor): ``[1, 1, R, 2*H*K + H*V]`` FLOAT32 TILE (R >= B): user b in row b; columns q (head-major
                ``[H, K]``), k (``[H, K]``), v (``[H, V]``).
            decay (ttnn.Tensor): ``[1, 1, R, H*K]`` FLOAT32 TILE per key-channel decay (exp applied).
            beta (ttnn.Tensor): ``[1, 1, R, H]`` FLOAT32 TILE (sigmoid applied).
            q_scale (float): query scale applied after the L2 norm.

        Keyword Args:
            eps (float): L2-norm epsilon. Defaults to ``1e-6``.

        Returns:
            ttnn.Tensor: o ``[1, 1, B*H, V]`` FLOAT32 TILE, row ``b*H + h``.
        )doc",
        &ttnn::experimental::kda::recurrent_decode,
        nb::arg("state").noconvert(),
        nb::arg("qkv").noconvert(),
        nb::arg("decay").noconvert(),
        nb::arg("beta").noconvert(),
        nb::arg("q_scale"),
        nb::kw_only(),
        nb::arg("eps") = 1e-6f);
}

}  // namespace ttnn::operations::experimental::kda::recurrent_decode::detail
