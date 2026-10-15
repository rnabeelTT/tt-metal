// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct MhcMixParams {
    uint32_t num_chunks;  // K chunks (split-K)
};

struct MhcMixTensorArgs {
    const Tensor& x;
    const Tensor& fn_t;
};

}  // namespace ttnn::experimental::prim
