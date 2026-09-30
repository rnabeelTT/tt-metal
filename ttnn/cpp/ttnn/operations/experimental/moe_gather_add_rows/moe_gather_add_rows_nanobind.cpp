// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_gather_add_rows_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/moe_gather_add_rows/moe_gather_add_rows.hpp"

namespace ttnn::operations::experimental::moe_gather_nb::detail {

void bind_experimental_moe_gather_add_rows_operation(nb::module_& mod) {
    const auto* doc =
        R"doc(
            MoE un-dispatch: out[t, :] = sum over the top_k slots j of rows[undispatch[t*top_k + j], :].

            Args:
                rows (ttnn.Tensor): bfloat16 ROW_MAJOR [Rd, H] (one stick per dispatched row).
                undispatch (ttnn.Tensor): uint32 ROW_MAJOR [1, T*top_k], destination row of every (token, slot) pair.
                num_tokens (int), top_k (int).

            Returns:
                bfloat16 TILE [1, 1, T, H].
        )doc";
    ttnn::bind_function<"moe_gather_add_rows">(
        mod,
        doc,
        &ttnn::operations::experimental::moe_gather_add_rows,
        nb::arg("rows").noconvert(),
        nb::arg("undispatch").noconvert(),
        nb::arg("num_tokens"),
        nb::arg("top_k"),
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::moe_gather_nb::detail
