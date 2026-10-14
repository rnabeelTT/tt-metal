// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental {

// Row scatter into a ROW_MAJOR cache, in place: for every r < R with row_idx[r] != 0xFFFFFFFF,
//     cache.row(row_idx[r]) = src.row(r)
// where row(i) is the i-th W-wide row of the tensor viewed as [rows, W] (W = last dim, one page per row).
// The indices are read on device, so the op is trace-safe (a replay writes wherever the index tensor points).
// Returns `cache`.
ttnn::Tensor scatter_rows(const Tensor& cache, const Tensor& src, const Tensor& row_idx);

}  // namespace ttnn::operations::experimental
