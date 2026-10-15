// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dsa_decode_scores_nanobind.hpp"

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/dsa_decode_scores/dsa_decode_scores.hpp"

namespace ttnn::operations::experimental::dsa_decode_scores_nb::detail {

void bind_experimental_dsa_decode_scores_operation(nb::module_& mod) {
    ttnn::bind_function<"dsa_decode_scores", "ttnn.experimental.">(
        mod,
        R"doc(
            Lightning-indexer scores of one decode query per slot over the slot's paged pooled-key cache (GLM-5 DSA):

                score[s, t] = sum_h relu(q[s, h, :] . k_s[t, :]) * weights[s, h]   (t < limits[s]; else -inf)

            with k_s[t] = k_pool.row(page_table[s][t // 32] * 32 + t % 32): 32-key page blocks, one tile row each.
            Blocks starting at or past limits[s] are neither read nor computed. Trace-safe (limits / page_table are
            read on device).

            Args:
                q (ttnn.Tensor): [1, 1, S*32, D] BFLOAT16 TILE, row s*32 + h = head h of slot s (32 heads).
                weights (ttnn.Tensor): [1, 1, S, 32] FLOAT32 TILE, row s = head weights of slot s.
                k_pool (ttnn.Tensor): BFLOAT16 TILE pooled-key pool, 32-key tile rows, last dim D.
                page_table (ttnn.Tensor): [rows >= S, max_blocks] int32 / uint32 ROW_MAJOR.
                limits (ttnn.Tensor): [1, S] uint32 ROW_MAJOR valid keys per slot.
                num_blocks (int): scored width in 32-key blocks.

            Returns:
                scores [1, 1, S, num_blocks * 32] FLOAT32 TILE.
        )doc",
        &ttnn::operations::experimental::dsa_decode_scores,
        nb::arg("q").noconvert(),
        nb::arg("weights").noconvert(),
        nb::arg("k_pool").noconvert(),
        nb::arg("page_table").noconvert(),
        nb::arg("limits").noconvert(),
        nb::arg("num_blocks"));
}

}  // namespace ttnn::operations::experimental::dsa_decode_scores_nb::detail
