// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/scatter_rows/scatter_rows.hpp"

#include "device/scatter_rows_device_operation.hpp"

namespace ttnn::operations::experimental {

ttnn::Tensor scatter_rows(const Tensor& cache, const Tensor& src, const Tensor& row_idx) {
    ttnn::prim::scatter_rows(cache, src, row_idx);
    return cache;
}

}  // namespace ttnn::operations::experimental
