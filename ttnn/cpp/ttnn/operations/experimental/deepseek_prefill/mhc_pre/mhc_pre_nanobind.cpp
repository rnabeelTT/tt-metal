// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_pre_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "mhc_pre.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::mhc_pre::detail {

void bind_experimental_mhc_pre_operation(nb::module_& mod) {
    ttnn::bind_function<"mhc_pre", "ttnn.experimental.deepseek_prefill.">(
        mod,
        R"doc(
            Fused mHC pre-collapse (Manifold-Constrained Hyper-Connections): the input of the wrapped sub-block,

                h = sum_i pre[:, i] * residual[:, i*C:(i+1)*C]

            over the packed [1,1,T,n*C] stream layout, in one pass (replaces n column slices of residual, n column
            slices of pre and n multiply / addcmul ops).

            Args:
                * :attr:`residual`: [1,1,T,n*C] FLOAT32 / BFLOAT16 TILE, stream i at columns [i*C,(i+1)*C).
                * :attr:`pre`: [1,1,T,n] TILE (same dtype), from mhc_split_sinkhorn.
                * :attr:`consts`: [n*n,32,32] TILE (same dtype) column-broadcast tiles (tile k has row k all ones),
                  the constants of mhc_post.
                * :attr:`n`: expansion rate (streams).
                * :attr:`output_dtype`: dtype of h (default: the input dtype); accumulated in fp32 DEST.

            Returns:
                h [1,1,T,C] TILE.
        )doc",
        &mhc_pre,
        nb::arg("residual").noconvert(),
        nb::arg("pre").noconvert(),
        nb::arg("consts").noconvert(),
        nb::arg("n"),
        nb::arg("output_dtype") = nb::none());
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::mhc_pre::detail
