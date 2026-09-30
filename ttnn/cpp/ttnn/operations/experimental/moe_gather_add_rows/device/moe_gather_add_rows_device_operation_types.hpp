// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

// MoE un-dispatch (Qwen3.5-122B autoport, doc/perf_pass turn 5): sum the k dispatched rows of every token. Replaces
// k x (slice + embedding gather + add) on the expert-sorted down-projection output.
struct MoeGatherAddRowsParams {
    uint32_t num_tokens;  // T
    uint32_t top_k;       // k
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct MoeGatherAddRowsInputs {
    Tensor rows;        // bfloat16 ROW_MAJOR [Rd, H]
    Tensor undispatch;  // uint32 ROW_MAJOR [1, T*k]
};

}  // namespace ttnn::experimental::prim
