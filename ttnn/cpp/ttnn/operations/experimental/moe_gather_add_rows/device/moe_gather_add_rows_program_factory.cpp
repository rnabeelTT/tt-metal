// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_gather_add_rows_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/hal.hpp>

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

MoeGatherAddRowsProgramFactory::cached_program_t MoeGatherAddRowsProgramFactory::create(
    const MoeGatherAddRowsParams& a, const MoeGatherAddRowsInputs& t, Tensor& out) {
    Program program{};
    const uint32_t T = a.num_tokens;
    const uint32_t K = a.top_k;
    const uint32_t H = t.rows.logical_shape()[-1];
    const uint32_t Wt = H / 32;
    const uint32_t RT = (T + 31) / 32;  // token tiles
    constexpr uint32_t tile_bytes = 32 * 32 * 2;

    // Work items = token tiles x column blocks; split the columns until there is one item per core (or the split
    // no longer divides Wt). A column block must stay small enough for 5 blocks of tiles in L1.
    const auto grid = t.rows.device()->compute_with_storage_grid_size();
    const uint32_t num_cores_max = grid.x * grid.y;
    uint32_t col_splits = 1;
    for (uint32_t cs : {1u, 2u, 3u, 4u, 6u, 8u, 12u, 16u}) {
        if (Wt % cs) {
            continue;
        }
        col_splits = cs;
        if (RT * cs >= num_cores_max) {
            break;
        }
    }
    while ((Wt / col_splits) * tile_bytes * 6 > 1'200'000 && Wt % (col_splits * 2) == 0) {
        col_splits *= 2;
    }
    const uint32_t Wt_c = Wt / col_splits;
    const uint32_t Hc = Wt_c * 32;
    const uint32_t num_items = RT * col_splits;

    auto [num_cores, all_cores, core_group_1, core_group_2, items_per_core_1, items_per_core_2] =
        split_work_to_cores(grid, num_items, true);

    auto make_cb = [&](uint32_t index, uint32_t pages, uint32_t page_size, tt::DataFormat fmt) {
        CreateCircularBuffer(
            program,
            all_cores,
            CircularBufferConfig(pages * page_size, {{index, fmt}}).set_page_size(index, page_size));
    };
    const uint32_t cb_in = tt::CBIndex::c_0, cb_til = tt::CBIndex::c_1, cb_acc_a = tt::CBIndex::c_2,
                   cb_acc_b = tt::CBIndex::c_3, cb_und = tt::CBIndex::c_4, cb_out = tt::CBIndex::c_5;
    make_cb(cb_in, 2 * Wt_c, tile_bytes, tt::DataFormat::Float16_b);  // row-major gathered block (double-buffered)
    make_cb(cb_til, Wt_c, tile_bytes, tt::DataFormat::Float16_b);
    make_cb(cb_acc_a, Wt_c, tile_bytes, tt::DataFormat::Float16_b);
    make_cb(cb_acc_b, Wt_c, tile_bytes, tt::DataFormat::Float16_b);
    make_cb(cb_out, Wt_c, tile_bytes, tt::DataFormat::Float16_b);  // the writer's buffer: never an add intermediate
    const uint32_t und_bytes = tt::align(32 * K * 4, hal::get_l1_alignment());
    make_cb(cb_und, 1, und_bytes, tt::DataFormat::UInt32);

    std::vector<uint32_t> reader_ct = {T, K, Wt_c, col_splits, Hc * 2, cb_in, cb_und};
    TensorAccessorArgs(*t.rows.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*t.undispatch.buffer()).append_to(reader_ct);
    std::vector<uint32_t> writer_ct = {Wt, Wt_c, col_splits, cb_out};
    TensorAccessorArgs(*out.buffer()).append_to(writer_ct);
    std::vector<uint32_t> compute_ct = {Wt_c, K, cb_in, cb_til, cb_acc_a, cb_acc_b, cb_out};

    const std::string kdir = "ttnn/cpp/ttnn/operations/experimental/moe_gather_add_rows/device/kernels/";
    auto reader = CreateKernel(
        program,
        kdir + "reader_moe_gather_add_rows.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_ct});
    auto writer = CreateKernel(
        program,
        kdir + "writer_moe_gather_add_rows.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_ct});
    auto compute = CreateKernel(
        program,
        kdir + "compute_moe_gather_add_rows.cpp",
        all_cores,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = false, .compile_args = compute_ct});

    std::vector<CoreCoord> cores;
    uint32_t item = 0;
    for (const auto& core : corerange_to_cores(all_cores, std::nullopt, true)) {
        const uint32_t n = core_group_1.contains(core) ? items_per_core_1 : items_per_core_2;
        SetRuntimeArgs(program, reader, core, {t.rows.buffer()->address(), t.undispatch.buffer()->address(), item, n});
        SetRuntimeArgs(program, writer, core, {out.buffer()->address(), item, n});
        SetRuntimeArgs(program, compute, core, {n});
        cores.push_back(core);
        item += n;
    }
    TT_FATAL(item == num_items, "moe_gather_add_rows: work split covered {} of {} items", item, num_items);
    return {std::move(program), shared_variables_t{reader, writer, cores}};
}

void MoeGatherAddRowsProgramFactory::override_runtime_arguments(
    cached_program_t& cached, const MoeGatherAddRowsParams&, const MoeGatherAddRowsInputs& t, Tensor& out) {
    auto& rd = GetRuntimeArgs(cached.program, cached.shared_variables.reader);
    auto& wr = GetRuntimeArgs(cached.program, cached.shared_variables.writer);
    for (const auto& core : cached.shared_variables.cores) {
        rd[core.x][core.y][0] = t.rows.buffer()->address();
        rd[core.x][core.y][1] = t.undispatch.buffer()->address();
        wr[core.x][core.y][0] = out.buffer()->address();
    }
}

}  // namespace ttnn::experimental::prim
