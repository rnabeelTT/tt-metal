// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

// Expert-sorted MoE dispatch bookkeeping in ONE single-core program (counting sort over the T*k (token, expert)
// pairs of a prefill segment): per dispatched row its source token and routing weight, per 32-row group its expert,
// per pair its destination row. Replaces ~100 small ttnn ops on a [T*k, E] one-hot (Qwen3.5-122B autoport,
// doc/perf_pass turn 5).
struct MoeSortedDispatchParams {
    uint32_t num_experts;      // E
    uint32_t top_k;            // k
    uint32_t group;            // rows per expert block (multiple of 32)
    uint32_t num_tokens;       // T
    uint32_t dispatched_rows;  // Rd (static upper bound, see moe_dispatch.dispatch_rows)
    uint32_t num_groups;       // G = Rd / group
    uint32_t pad_group_id;     // expert id written for padding groups (0, or a sentinel >= E the grouped matmul skips)
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct MoeSortedDispatchInputs {
    Tensor ids;     // uint32 ROW_MAJOR [1, T*k]: expert of every (token, slot) pair, token-major
    Tensor scores;  // bfloat16 ROW_MAJOR [1, T*k]: routing weight of every pair
};

}  // namespace ttnn::experimental::prim
