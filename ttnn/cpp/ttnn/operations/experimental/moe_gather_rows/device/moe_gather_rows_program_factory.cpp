// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "moe_gather_rows_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/hal.hpp>

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

// Work item = (group g, column block cs). The reader gathers the group's 32 source rows (columns of block cs) into a
// row-major 32 x Hc block (zeros, without any NoC read, for a padding group), the compute kernel tilizes it, the
// writer stores the Wt_c tiles at the group's tile offset - or drops them when the reader flagged a padding group.
MoeGatherRowsProgramFactory::cached_program_t MoeGatherRowsProgramFactory::create(
    const MoeGatherRowsParams& a, const MoeGatherRowsInputs& t, Tensor& out) {
    Program program{};
    const uint32_t H = t.rows.logical_shape()[-1];
    const uint32_t G = t.group_ids.logical_shape()[-1];
    const uint32_t Wt = H / 32;
    constexpr uint32_t tile_bytes = 32 * 32 * 2;

    const auto grid = t.rows.device()->compute_with_storage_grid_size();
    const uint32_t num_cores_max = grid.x * grid.y;
    uint32_t col_splits = 1;
    for (uint32_t cs : {1u, 2u, 3u, 4u, 6u, 8u, 12u, 16u}) {
        if (Wt % cs) {
            continue;
        }
        col_splits = cs;
        if (G * cs >= num_cores_max) {
            break;
        }
    }
    // the gathered block (double-buffered) + the tilized block must fit L1
    while ((Wt / col_splits) * tile_bytes * 4 > 1'200'000 && Wt % (col_splits * 2) == 0) {
        col_splits *= 2;
    }
    const uint32_t Wt_c = Wt / col_splits;
    const uint32_t Hc = Wt_c * 32;
    const uint32_t num_items = G * col_splits;
    auto [num_cores, all_cores, core_group_1, core_group_2, items_per_core_1, items_per_core_2] =
        split_work_to_cores(grid, num_items, true);

    auto make_cb = [&](uint32_t index, uint32_t pages, uint32_t page_size, tt::DataFormat fmt) {
        CreateCircularBuffer(
            program,
            all_cores,
            CircularBufferConfig(pages * page_size, {{index, fmt}}).set_page_size(index, page_size));
    };
    const uint32_t cb_in = tt::CBIndex::c_0, cb_out = tt::CBIndex::c_1, cb_src = tt::CBIndex::c_2,
                   cb_gid = tt::CBIndex::c_3, cb_flag = tt::CBIndex::c_4;
    make_cb(cb_in, 2 * Wt_c, tile_bytes, tt::DataFormat::Float16_b);   // row-major gathered block (double-buffered)
    make_cb(cb_out, 2 * Wt_c, tile_bytes, tt::DataFormat::Float16_b);  // tilized block
    const uint32_t src_bytes = tt::align(32 * 4, hal::get_l1_alignment());
    make_cb(cb_src, 1, src_bytes, tt::DataFormat::UInt32);
    const uint32_t gid_bytes = tt::align(G * 2, hal::get_l1_alignment());
    make_cb(cb_gid, 1, gid_bytes, tt::DataFormat::UInt16);
    const uint32_t flag_bytes = tt::align(4, hal::get_l1_alignment());
    make_cb(cb_flag, 4, flag_bytes, tt::DataFormat::UInt32);  // reader -> writer: 1 = real group, 0 = padding

    std::vector<uint32_t> reader_ct = {a.num_experts, Wt_c, col_splits, Hc * 2, G, cb_in, cb_src, cb_gid, cb_flag};
    TensorAccessorArgs(*t.rows.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*t.src_tok.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*t.group_ids.buffer()).append_to(reader_ct);
    std::vector<uint32_t> writer_ct = {Wt, Wt_c, col_splits, cb_out, cb_flag};
    TensorAccessorArgs(*out.buffer()).append_to(writer_ct);
    std::vector<uint32_t> compute_ct = {Wt_c, cb_in, cb_out};

    const std::string kdir = "ttnn/cpp/ttnn/operations/experimental/moe_gather_rows/device/kernels/";
    auto reader = CreateKernel(
        program,
        kdir + "reader_moe_gather_rows.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_ct});
    auto writer = CreateKernel(
        program,
        kdir + "writer_moe_gather_rows.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_ct});
    auto compute = CreateKernel(
        program,
        kdir + "compute_moe_gather_rows.cpp",
        all_cores,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = false, .compile_args = compute_ct});

    std::vector<CoreCoord> cores;
    uint32_t item = 0;
    for (const auto& core : corerange_to_cores(all_cores, std::nullopt, true)) {
        const uint32_t n = core_group_1.contains(core) ? items_per_core_1 : items_per_core_2;
        SetRuntimeArgs(
            program,
            reader,
            core,
            {t.rows.buffer()->address(), t.src_tok.buffer()->address(), t.group_ids.buffer()->address(), item, n});
        SetRuntimeArgs(program, writer, core, {out.buffer()->address(), item, n});
        SetRuntimeArgs(program, compute, core, {n});
        cores.push_back(core);
        item += n;
    }
    TT_FATAL(item == num_items, "moe_gather_rows: work split covered {} of {} items", item, num_items);
    return {std::move(program), shared_variables_t{reader, writer, cores}};
}

void MoeGatherRowsProgramFactory::override_runtime_arguments(
    cached_program_t& cached, const MoeGatherRowsParams&, const MoeGatherRowsInputs& t, Tensor& out) {
    auto& rd = GetRuntimeArgs(cached.program, cached.shared_variables.reader);
    auto& wr = GetRuntimeArgs(cached.program, cached.shared_variables.writer);
    for (const auto& core : cached.shared_variables.cores) {
        rd[core.x][core.y][0] = t.rows.buffer()->address();
        rd[core.x][core.y][1] = t.src_tok.buffer()->address();
        rd[core.x][core.y][2] = t.group_ids.buffer()->address();
        wr[core.x][core.y][0] = out.buffer()->address();
    }
}

}  // namespace ttnn::experimental::prim
