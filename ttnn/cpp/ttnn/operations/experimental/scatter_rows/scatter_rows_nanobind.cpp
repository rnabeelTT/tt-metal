// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "scatter_rows_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/scatter_rows/scatter_rows.hpp"

namespace ttnn::operations::experimental::scatter_rows_nb::detail {

void bind_experimental_scatter_rows_operation(nb::module_& mod) {
    const auto* doc =
        R"doc(
            Row scatter into a ROW_MAJOR cache, in place: for every r with row_idx[r] != 0xFFFFFFFF,
            cache.row(row_idx[r]) = src.row(r). Rows are the pages of the tensors viewed as [rows, W]
            (W = last dim). The indices are read on device (trace-safe).

            Args:
                cache (ttnn.Tensor): ROW_MAJOR interleaved, written in place.
                src (ttnn.Tensor): ROW_MAJOR, same dtype and last dim, R rows.
                row_idx (ttnn.Tensor): uint32 / int32 ROW_MAJOR one stick [.., R].

            Returns:
                cache.
        )doc";

    ttnn::bind_function<"scatter_rows">(
        mod,
        doc,
        &ttnn::operations::experimental::scatter_rows,
        nb::arg("cache").noconvert(),
        nb::arg("src").noconvert(),
        nb::arg("row_idx").noconvert());
}

}  // namespace ttnn::operations::experimental::scatter_rows_nb::detail
