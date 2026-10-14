// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct ScatterRowsParams {};

struct ScatterRowsInputs {
    Tensor cache;    // ROW_MAJOR, interleaved; written in place
    Tensor src;      // ROW_MAJOR, same dtype and last dim as cache: R rows
    Tensor row_idx;  // uint32 ROW_MAJOR one stick [.., R]; 0xFFFFFFFF = skip
};

}  // namespace ttnn::experimental::prim
