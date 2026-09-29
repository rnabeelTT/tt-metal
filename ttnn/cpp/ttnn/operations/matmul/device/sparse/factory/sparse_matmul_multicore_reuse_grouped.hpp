// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation_types.hpp"

namespace ttnn::prim {

// Group-parallel sparse matmul for the indexed/gather mode (`indices` operand present, in1 sparse):
//
//   a       [.., G, M, K]   compact activation groups (group g = the rows routed to expert indices[g])
//   b       [.., E, K, N]   expert weights, one [K, N] block per expert
//   indices [1, G] uint16   expert id of every group (device-resident, read by the kernels at run time)
//   out     [1, G, M, N]    compact
//
// The multicast 1D factory processes every group on every core and only splits N across cores, so an
// MoE with a small per-device N (tensor-parallel experts) leaves most of the grid idle. This factory
// splits the GROUP axis across the cores instead (`split_work_to_cores` over G whole groups; each core
// reads its own groups' activations and the whole [K, N] block of each group's expert from DRAM, no
// multicast), using the plain batched-matmul kernels (reader_bmm_tile_layout_in0 +
// reader_writer_bmm_tile_layout_in1_indexed + bmm_large_block_zm_fused_bias_activation).
//
// Selected by passing a MatmulMultiCoreReuseProgramConfig with per_core_M == Mt and per_core_N == Nt
// (whole groups per core). Numerics are the same K-blocked accumulation as the multicast factory.
struct SparseMatmulMultiCoreReuseGroupedProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle in0_reader_kernel{};
        tt::tt_metal::KernelHandle in1_writer_kernel{};
        std::vector<CoreCoord> cores;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const ttnn::prim::SparseMatmulParams& operation_attributes,
        const ttnn::prim::SparseMatmulInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const ttnn::prim::SparseMatmulParams& operation_attributes,
        const ttnn::prim::SparseMatmulInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);
};

}  // namespace ttnn::prim
