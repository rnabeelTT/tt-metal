// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_mix_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "mhc_mix.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::mhc_mix::detail {

void bind_experimental_mhc_mix_operation(nb::module_& mod) {
    ttnn::bind_function<"mhc_mix", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            Split-K partials of the mHC mix projection and of the row sums of squares over the packed stream:
            partials[0, c, :, 0:32] = x[:, chunk c] @ fn_t[chunk c, :]; partials[0, c, :, 32:64] = the per-row-tile
            Gram x[:, chunk c] @ x[:, chunk c]^T (diagonal = the chunk's row sum of squares). One (row tile, chunk) unit
            per core; the caller sums over the chunks and scales by rsqrt(mean(x^2) + eps).

            Args:
                * :attr:`x`: [1, 1, T, K] FLOAT32 TILE.
                * :attr:`fn_t`: [K, 32] FLOAT32 TILE (projection, zero padded to 32 columns).
                * :attr:`num_chunks`: K chunks (must divide K / 32).

            Returns:
                partials [1, num_chunks, T, 64] FLOAT32 TILE.
        )doc",
        &mhc_mix,
        nb::arg("x").noconvert(),
        nb::arg("fn_t").noconvert(),
        nb::arg("num_chunks"));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::mhc_mix::detail
