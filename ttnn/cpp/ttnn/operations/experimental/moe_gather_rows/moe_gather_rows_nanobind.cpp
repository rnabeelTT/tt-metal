// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "moe_gather_rows_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/moe_gather_rows/moe_gather_rows.hpp"

namespace ttnn::operations::experimental::moe_gather_rows_nb::detail {

void bind_experimental_moe_gather_rows_operation(nb::module_& mod) {
    const auto* doc =
        R"doc(
            MoE dispatch gather: out[g, r, :] = rows[src_tok[g*32 + r], :] for every real group g
            (group_ids[g] < num_experts), written directly as tiles; padding groups are skipped.

            Args:
                rows (ttnn.Tensor): bfloat16 ROW_MAJOR [T, H] (one stick per token).
                src_tok (ttnn.Tensor): uint32 ROW_MAJOR [1, G*32], source token of every dispatched row.
                group_ids (ttnn.Tensor): uint16 ROW_MAJOR [1, G], expert id of every group (>= num_experts = padding).
                num_experts (int).

            Returns:
                bfloat16 TILE [1, G, 32, H] (padding groups' tiles are not written).
        )doc";

    ttnn::bind_function<"moe_gather_rows">(
        mod,
        doc,
        &ttnn::operations::experimental::moe_gather_rows,
        nb::arg("rows").noconvert(),
        nb::arg("src_tok").noconvert(),
        nb::arg("group_ids").noconvert(),
        nb::arg("num_experts"),
        nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::moe_gather_rows_nb::detail
