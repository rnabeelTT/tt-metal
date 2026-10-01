// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct MoeGatherRowsParams {
    uint32_t num_experts;  // E: group ids >= E are padding groups (skipped)
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct MoeGatherRowsInputs {
    Tensor rows;       // bfloat16 ROW_MAJOR [T, H]
    Tensor src_tok;    // uint32 ROW_MAJOR [1, G*32]
    Tensor group_ids;  // uint16 ROW_MAJOR [1, G]
};

}  // namespace ttnn::experimental::prim
