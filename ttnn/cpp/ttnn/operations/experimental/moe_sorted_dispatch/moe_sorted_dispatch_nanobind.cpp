// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_sorted_dispatch_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/moe_sorted_dispatch/moe_sorted_dispatch.hpp"

namespace ttnn::operations::experimental::moe_dispatch_nb::detail {

void bind_experimental_moe_sorted_dispatch_operation(nb::module_& mod) {
    const auto* doc =
        R"doc(
            Expert-sorted MoE dispatch bookkeeping in one single-core program (integer counting sort over the T*k
            (token, slot) pairs of a prefill segment).

            Args:
                ids (ttnn.Tensor): uint32 ROW_MAJOR [1, T*k], expert of every pair, token-major (ids >= num_experts are ignored).
                scores (ttnn.Tensor): bfloat16 ROW_MAJOR [1, T*k], routing weight of every pair.
                num_experts (int), top_k (int), group (int, rows per expert block, multiple of 32),
                dispatched_rows (int): static padded row count (>= T*k + num_experts*(group-1) + group).

            Returns:
                (src_tok uint32 [1, Rd], weights bfloat16 [1, Rd], group_ids uint16 [1, Rd/group], undispatch uint32 [1, T*k]).
        )doc";
    ttnn::bind_function<"moe_sorted_dispatch">(
        mod,
        doc,
        &ttnn::operations::experimental::moe_sorted_dispatch,
        nb::arg("ids").noconvert(),
        nb::arg("scores").noconvert(),
        nb::arg("num_experts"),
        nb::arg("top_k"),
        nb::arg("group"),
        nb::arg("dispatched_rows"),
        nb::arg("memory_config") = nb::none(),
        nb::arg("pad_group_id") = 0);
}

}  // namespace ttnn::operations::experimental::moe_dispatch_nb::detail
