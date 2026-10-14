// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "scatter_rows_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/hal.hpp>

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

// Work item = one source row. Each core reads its slice of the index stick once, then for every row reads the source
// page into L1 and writes it to the cache page the index names (index 0xFFFFFFFF = skip). One data-movement kernel.
ScatterRowsProgramFactory::cached_program_t ScatterRowsProgramFactory::create(
    const ScatterRowsParams&, const ScatterRowsInputs& t, Tensor& /*out*/) {
    Program program{};
    const uint32_t W = t.src.logical_shape()[-1];
    const uint32_t R = t.src.logical_volume() / W;
    const uint32_t row_bytes = t.src.buffer()->page_size();
    TT_FATAL(row_bytes == t.cache.buffer()->page_size(), "scatter_rows: cache and src page sizes differ");

    const auto grid = t.src.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, rows_per_core_1, rows_per_core_2] =
        split_work_to_cores(grid, R, true);

    const uint32_t cb_row = tt::CBIndex::c_0, cb_idx = tt::CBIndex::c_1;
    const uint32_t row_cb_bytes = tt::align(row_bytes, hal::get_l1_alignment());
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(2 * row_cb_bytes, {{cb_row, tt::DataFormat::UInt8}}).set_page_size(cb_row, row_cb_bytes));
    // every core reads the whole index stick (R * 4 bytes) and walks its own slice
    const uint32_t idx_stick_bytes = t.row_idx.buffer()->page_size();
    TT_FATAL(idx_stick_bytes >= R * 4, "scatter_rows: row_idx must be a single page holding all R indices");
    const uint32_t idx_bytes = tt::align(idx_stick_bytes, hal::get_l1_alignment());
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(idx_bytes, {{cb_idx, tt::DataFormat::UInt32}}).set_page_size(cb_idx, idx_bytes));

    std::vector<uint32_t> ct = {row_bytes, cb_row, cb_idx, (uint32_t)hal::get_l1_alignment(), idx_stick_bytes};
    TensorAccessorArgs(*t.src.buffer()).append_to(ct);
    TensorAccessorArgs(*t.cache.buffer()).append_to(ct);
    TensorAccessorArgs(*t.row_idx.buffer()).append_to(ct);
    auto kernel = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/scatter_rows/device/kernels/scatter_rows.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = ct});

    std::vector<CoreCoord> cores;
    uint32_t row = 0;
    for (const auto& core : corerange_to_cores(all_cores, std::nullopt, true)) {
        const uint32_t n = core_group_1.contains(core) ? rows_per_core_1 : rows_per_core_2;
        SetRuntimeArgs(
            program,
            kernel,
            core,
            {t.src.buffer()->address(), t.cache.buffer()->address(), t.row_idx.buffer()->address(), row, n});
        cores.push_back(core);
        row += n;
    }
    TT_FATAL(row == R, "scatter_rows: work split covered {} of {} rows", row, R);
    return {std::move(program), shared_variables_t{kernel, cores}};
}

void ScatterRowsProgramFactory::override_runtime_arguments(
    cached_program_t& cached, const ScatterRowsParams&, const ScatterRowsInputs& t, Tensor&) {
    auto& args = GetRuntimeArgs(cached.program, cached.shared_variables.kernel);
    for (const auto& core : cached.shared_variables.cores) {
        args[core.x][core.y][0] = t.src.buffer()->address();
        args[core.x][core.y][1] = t.cache.buffer()->address();
        args[core.x][core.y][2] = t.row_idx.buffer()->address();
    }
}

}  // namespace ttnn::experimental::prim
