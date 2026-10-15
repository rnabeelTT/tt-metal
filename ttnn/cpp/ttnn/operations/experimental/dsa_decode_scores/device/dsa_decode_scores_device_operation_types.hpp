// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct DsaDecodeScoresParams {
    uint32_t num_blocks;  // scored width in 32-key blocks
};

struct DsaDecodeScoresTensorArgs {
    const Tensor& q;
    const Tensor& weights;
    const Tensor& k_pool;
    const Tensor& page_table;
    const Tensor& limits;
};

}  // namespace ttnn::experimental::prim
