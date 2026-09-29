// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_grouped.hpp"

#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"

namespace ttnn::prim {

using namespace tt;
using namespace operations::matmul::utilities;

SparseMatmulMultiCoreReuseGroupedProgramFactory::cached_program_t
SparseMatmulMultiCoreReuseGroupedProgramFactory::create(
    const ttnn::prim::SparseMatmulParams& operation_attributes,
    const ttnn::prim::SparseMatmulInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    tt_metal::Program program{};

    const auto& a = tensor_args.input_tensors.at(0);
    const auto& b = tensor_args.input_tensors.at(1);
    const auto& output_tensor = tensor_return_value.at(0);

    TT_FATAL(
        operation_attributes.use_indices && !tensor_args.optional_input_tensors.empty() &&
            tensor_args.optional_input_tensors.at(0).has_value(),
        "sparse_matmul (grouped): the group-parallel factory needs the `indices` operand (indexed/gather mode)");
    const Tensor& indices = tensor_args.optional_input_tensors.at(0).value();
    TT_FATAL(
        operation_attributes.is_input_b_sparse && !operation_attributes.is_input_a_sparse,
        "sparse_matmul (grouped): input A must be the compact group activations (dense) and input B the expert "
        "weights (sparse)");
    TT_FATAL(operation_attributes.program_config.has_value(), "sparse_matmul (grouped): program_config required");
    const auto* pc = std::get_if<operations::matmul::MatmulMultiCoreReuseProgramConfig>(
        &operation_attributes.program_config.value());
    TT_FATAL(pc != nullptr, "sparse_matmul (grouped): expected a MatmulMultiCoreReuseProgramConfig");

    const uint32_t in0_block_w = pc->in0_block_w;
    const uint32_t out_subblock_h = pc->out_subblock_h;
    const uint32_t out_subblock_w = pc->out_subblock_w;
    const uint32_t per_core_M = pc->per_core_M;
    const uint32_t per_core_N = pc->per_core_N;

    // ---- shapes
    const auto& ashape = get_matmul_tensor_padded_shape(a, /*transpose=*/false);
    const auto& bshape = get_matmul_tensor_padded_shape(b, /*transpose=*/false);
    const auto in0_tile = get_matmul_tile(a, /*transpose=*/false);
    const auto in1_tile = get_matmul_tile(b, /*transpose=*/false);
    const auto output_tile = tt::tt_metal::Tile({in0_tile.get_height(), in1_tile.get_width()});

    const uint32_t G = static_cast<uint32_t>(indices.logical_volume());  // groups = output batch
    const uint32_t E = bshape[-3];                                       // experts = in1 sparse groups
    const uint32_t Mt = get_M_dim(ashape, in0_tile, /*fuse_batch=*/false);
    const uint32_t Kt = get_K_dim(ashape, in0_tile);
    const uint32_t Nt = get_N_dim(bshape, in1_tile);
    TT_FATAL(
        get_batch_size(ashape) == G,
        "sparse_matmul (grouped): A batch ({}) must equal the number of indices ({})",
        get_batch_size(ashape),
        G);
    TT_FATAL(
        per_core_M == Mt,
        "sparse_matmul (grouped): per_core_M ({}) must equal Mt ({}) (whole groups per core)",
        per_core_M,
        Mt);
    TT_FATAL(
        per_core_N == Nt,
        "sparse_matmul (grouped): per_core_N ({}) must equal Nt ({}) (whole groups per core)",
        per_core_N,
        Nt);
    TT_FATAL(
        Kt % in0_block_w == 0,
        "sparse_matmul (grouped): Kt ({}) must be divisible by in0_block_w ({})",
        Kt,
        in0_block_w);
    TT_FATAL(out_subblock_h != 0 && out_subblock_w != 0, "sparse_matmul (grouped): subblock dims must be non-zero");
    TT_FATAL(
        Mt % out_subblock_h == 0,
        "sparse_matmul (grouped): Mt ({}) must be divisible by out_subblock_h ({})",
        Mt,
        out_subblock_h);
    TT_FATAL(
        Nt % out_subblock_w == 0,
        "sparse_matmul (grouped): Nt ({}) must be divisible by out_subblock_w ({})",
        Nt,
        out_subblock_w);
    const auto out_shape = output_tensor.padded_shape();
    TT_FATAL(
        out_shape.rank() == 4 && out_shape[0] == 1 && out_shape[1] == G &&
            out_shape[-2] / output_tile.get_height() == Mt && out_shape[-1] / output_tile.get_width() == Nt,
        "sparse_matmul (grouped): output must be the compact [1, G, M, N] tensor (G={}, Mt={}, Nt={}), got {}",
        G,
        Mt,
        Nt,
        out_shape);

    // ---- data formats / tile sizes
    const auto in0_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    const auto in1_data_format = tt_metal::datatype_to_dataformat_converter(b.dtype());
    const auto output_data_format = tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    auto* const device = a.device();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config.value());
    if (fp32_dest_acc_en) {
        TT_FATAL(
            out_subblock_h * out_subblock_w <= 4,
            "sparse_matmul (grouped): a subblock may hold at most 4 tiles with fp32 dest accumulation");
    }
    const uint32_t num_blocks = Kt / in0_block_w;
    const bool packer_l1_acc_en = packer_l1_acc && (num_blocks > 2);
    const tt::DataFormat interm0_data_format =
        packer_l1_acc_en ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                         : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);

    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    const uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    const uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    const uint32_t in0_aligned_tile_size = tt::align(in0_single_tile_size, dram_alignment);
    const uint32_t in1_aligned_tile_size = tt::align(in1_single_tile_size, dram_alignment);
    const uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);
    const uint32_t interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

    auto* const in0_buffer = a.buffer();
    auto* const in1_buffer = b.buffer();
    auto* const idx_buffer = indices.buffer();
    auto* const out_buffer = output_tensor.buffer();
    TT_FATAL(
        !a.is_sharded() && !b.is_sharded() && !output_tensor.is_sharded(),
        "sparse_matmul (grouped): interleaved operands only");

    // ---- CB sizes (double-buffered inputs, one output block)
    const uint32_t in0_block_num_tiles = per_core_M * in0_block_w;
    const uint32_t in0_CB_size = 2 * in0_block_num_tiles * in0_aligned_tile_size;
    const uint32_t in1_block_num_tiles = per_core_N * in0_block_w;
    const uint32_t in1_CB_size = 2 * in1_block_num_tiles * in1_aligned_tile_size;
    const uint32_t out_block_tiles = per_core_M * per_core_N;
    const uint32_t out_CB_size = out_block_tiles * output_single_tile_size;
    const uint32_t interm0_CB_size = out_block_tiles * interm0_single_tile_size;
    const uint32_t idx_page_size = idx_buffer->aligned_page_size();

    // ---- core split over whole groups
    const CoreCoord grid = pc->compute_with_storage_grid_size;
    uint32_t num_cores = 0, groups_g1 = 0, groups_g2 = 0;
    CoreRangeSet all_cores, core_group_1, core_group_2;
    if (pc->allowed_worker_cores.has_value()) {
        std::tie(num_cores, all_cores, core_group_1, core_group_2, groups_g1, groups_g2) =
            tt::tt_metal::split_work_to_cores(pc->allowed_worker_cores.value(), G, /*row_wise=*/true);
    } else {
        std::tie(num_cores, all_cores, core_group_1, core_group_2, groups_g1, groups_g2) =
            tt::tt_metal::split_work_to_cores(grid, G, /*row_wise=*/true);
    }
    TT_FATAL(
        num_cores > 0 && groups_g1 > 0, "sparse_matmul (grouped): no work split (G={}, grid={}x{})", G, grid.x, grid.y);
    // Row-wise split above and row-wise enumeration here: every core's runtime batch (below) must agree with
    // the compute kernel it was built with (core_group_1 vs core_group_2), so membership is checked per core.
    const auto cores = corerange_to_cores(all_cores, num_cores, /*row_wise=*/true);

    // ---- compile-time args
    const uint32_t in0_tensor_stride_w = 1;
    const uint32_t in0_tensor_stride_h = Kt;
    const uint32_t in0_tensor_next_block_stride = in0_block_w;
    const uint32_t in1_tensor_stride_w = 1;
    const uint32_t in1_tensor_stride_h = Nt;
    const uint32_t in1_tensor_next_block_stride = in0_block_w * Nt;
    const auto a_shape_logical = get_matmul_tensor_logical_shape(a, /*transpose=*/false);
    const uint32_t in0_last_ktile_w = a_shape_logical[-1] % in0_tile.get_width();

    std::vector<uint32_t> reader_compile_time_args = {
        in0_tensor_stride_w,
        in0_tensor_stride_h,
        in0_tensor_next_block_stride,
        in0_block_w,
        per_core_M,           // in0_block_h
        in0_block_num_tiles,  // in0_block_num_tiles
        in0_last_ktile_w,
        0u,          // in0_last_ktile_h
        num_blocks,  // num_blocks
        0u,          // bcast_B (unused by the in0 reader)
        Mt * Kt,     // MtKt: compact group stride in in0
    };
    tt::tt_metal::TensorAccessorArgs(*in0_buffer).append_to(reader_compile_time_args);

    const uint32_t in1_num_subblocks = per_core_N / out_subblock_w;
    const uint32_t out_num_subblocks_h = per_core_M / out_subblock_h;
    std::vector<uint32_t> writer_compile_time_args = {
        in1_tensor_stride_w,
        in1_tensor_stride_h,
        in1_tensor_next_block_stride,
        per_core_N,           // in1_block_w
        in0_block_w,          // in1_block_h
        in1_block_num_tiles,  // in1_block_num_tiles
        num_blocks,
        0u,                   // bcast_B (unused)
        Kt * Nt,              // KtNt: expert stride in in1
        1u,                   // out_tensor_stride_w
        Nt,                   // out_tensor_stride_h
        out_subblock_w,       // out_tensor_next_subblock_stride_w
        out_subblock_h * Nt,  // out_tensor_next_subblock_stride_h
        out_subblock_w,
        out_subblock_h,
        out_subblock_w * out_subblock_h,
        in1_num_subblocks,    // out_num_subblocks_w
        out_num_subblocks_h,  // out_num_subblocks_h
        Mt * Nt,              // MtNt: compact group stride in out
        idx_page_size,        // idx_page_bytes
        G,                    // num_groups_total
        E,                    // num_experts
    };
    tt::tt_metal::TensorAccessorArgs(*in1_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*out_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*idx_buffer).append_to(writer_compile_time_args);

    // ---- kernels
    const std::unordered_map<std::string, uint32_t> reader_named_args = {{"cb_in0", tt::CBIndex::c_0}};
    const std::unordered_map<std::string, uint32_t> writer_named_args = {
        {"cb_in1", tt::CBIndex::c_1}, {"cb_out", tt::CBIndex::c_4}, {"cb_idx", tt::CBIndex::c_7}};

    auto in0_reader_kernel = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args,
            .named_compile_args = reader_named_args});
    auto in1_writer_kernel = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_writer_bmm_tile_layout_in1_indexed.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args,
            .named_compile_args = writer_named_args});

    std::map<std::string, std::string> mm_kernel_defines;
    mm_kernel_defines["FUSE_ACTIVATION"] = "0";
    if (packer_l1_acc_en) {
        mm_kernel_defines["PACKER_L1_ACC"] = "1";
    }
    if (fp32_dest_acc_en) {
        mm_kernel_defines["FP32_DEST_ACC_EN"] = "1";
    }
    ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
        device->arch(), num_cores, mm_kernel_defines);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(),
        num_cores,
        mm_kernel_defines,
        ttnn::get_throttle_level(operation_attributes.compute_kernel_config));

    const uint32_t in0_num_subblocks = per_core_M / out_subblock_h;
    const uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;
    const uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;
    const uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;
    auto compute_args_for = [&](uint32_t groups_per_core) {
        return std::vector<uint32_t>{
            in0_block_w,
            in0_num_subblocks,
            in0_block_num_tiles,
            in0_subblock_num_tiles,
            in1_num_subblocks,
            in1_block_num_tiles,
            in1_per_core_w,
            num_blocks,
            1u,  // out_num_blocks_x
            1u,  // out_num_blocks_y
            out_subblock_h,
            out_subblock_w,
            out_subblock_num_tiles,
            groups_per_core,  // batch
            out_block_tiles,
            0u,  // untilize_out
            0u,  // get_batch_from_reader
            0u,  // in0_transpose_tile
        };
    };
    const std::unordered_map<std::string, uint32_t> compute_named_args = {
        {"cb_in0", tt::CBIndex::c_0},
        {"cb_in1", tt::CBIndex::c_1},
        {"cb_bias", tt::CBIndex::c_3},
        {"cb_out", tt::CBIndex::c_4},
        {"cb_intermed0", tt::CBIndex::c_5},
        {"cb_in0_transposed", tt::CBIndex::c_10},
    };
    tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp",
        core_group_1,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_args_for(groups_g1),
            .defines = mm_kernel_defines,
            .named_compile_args = compute_named_args});
    if (core_group_2.num_cores() > 0) {
        tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp",
            core_group_2,
            tt_metal::ComputeConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .dst_full_sync_en = dst_full_sync_en,
                .math_approx_mode = math_approx_mode,
                .compile_args = compute_args_for(groups_g2),
                .defines = mm_kernel_defines,
                .named_compile_args = compute_named_args});
    }

    // ---- circular buffers
    tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt_metal::CircularBufferConfig(in0_CB_size, {{tt::CBIndex::c_0, in0_data_format}})
            .set_page_size(tt::CBIndex::c_0, in0_aligned_tile_size)
            .set_tile_dims(tt::CBIndex::c_0, in0_tile));
    tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt_metal::CircularBufferConfig(in1_CB_size, {{tt::CBIndex::c_1, in1_data_format}})
            .set_page_size(tt::CBIndex::c_1, in1_aligned_tile_size)
            .set_tile_dims(tt::CBIndex::c_1, in1_tile));
    tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt_metal::CircularBufferConfig(
            idx_page_size, {{tt::CBIndex::c_7, tt_metal::datatype_to_dataformat_converter(indices.dtype())}})
            .set_page_size(tt::CBIndex::c_7, idx_page_size));
    if (interm0_data_format != output_data_format) {
        tt_metal::CreateCircularBuffer(
            program,
            all_cores,
            tt_metal::CircularBufferConfig(out_CB_size, {{tt::CBIndex::c_4, output_data_format}})
                .set_page_size(tt::CBIndex::c_4, output_single_tile_size)
                .set_tile_dims(tt::CBIndex::c_4, output_tile));
        tt_metal::CreateCircularBuffer(
            program,
            all_cores,
            tt_metal::CircularBufferConfig(interm0_CB_size, {{tt::CBIndex::c_5, interm0_data_format}})
                .set_page_size(tt::CBIndex::c_5, interm0_single_tile_size)
                .set_tile_dims(tt::CBIndex::c_5, output_tile));
    } else {
        std::map<uint8_t, tt::DataFormat> spec{
            {tt::CBIndex::c_4, output_data_format}, {tt::CBIndex::c_5, interm0_data_format}};
        tt_metal::CreateCircularBuffer(
            program,
            all_cores,
            tt_metal::CircularBufferConfig(out_CB_size, spec)
                .set_page_size(tt::CBIndex::c_4, output_single_tile_size)
                .set_page_size(tt::CBIndex::c_5, interm0_single_tile_size)
                .set_tile_dims(tt::CBIndex::c_4, output_tile)
                .set_tile_dims(tt::CBIndex::c_5, output_tile));
    }

    // ---- runtime args: consecutive whole groups per core
    uint32_t groups_written = 0;
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const auto& core = cores[i];
        const uint32_t groups_here = core_group_1.contains(core) ? groups_g1 : groups_g2;
        tt_metal::SetRuntimeArgs(
            program, in0_reader_kernel, core, {in0_buffer->address(), groups_written * Mt * Kt, groups_here});
        tt_metal::SetRuntimeArgs(
            program,
            in1_writer_kernel,
            core,
            {in1_buffer->address(),
             groups_here,
             out_buffer->address(),
             groups_written * Mt * Nt,
             idx_buffer->address(),
             groups_written});
        groups_written += groups_here;
    }
    TT_FATAL(groups_written == G, "sparse_matmul (grouped): distributed {} of {} groups", groups_written, G);

    return {std::move(program), shared_variables_t{in0_reader_kernel, in1_writer_kernel, cores}};
}

void SparseMatmulMultiCoreReuseGroupedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const ttnn::prim::SparseMatmulParams& /*operation_attributes*/,
    const ttnn::prim::SparseMatmulInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    auto& program = cached_program.program;
    const auto& sv = cached_program.shared_variables;
    const auto in0_addr = tensor_args.input_tensors.at(0).buffer()->address();
    const auto in1_addr = tensor_args.input_tensors.at(1).buffer()->address();
    const auto idx_addr = tensor_args.optional_input_tensors.at(0)->buffer()->address();
    const auto out_addr = tensor_return_value.at(0).buffer()->address();
    auto& reader_args_by_core = GetRuntimeArgs(program, sv.in0_reader_kernel);
    auto& writer_args_by_core = GetRuntimeArgs(program, sv.in1_writer_kernel);
    for (const auto& core : sv.cores) {
        auto& r = reader_args_by_core[core.x][core.y];
        r[0] = in0_addr;
        auto& w = writer_args_by_core[core.x][core.y];
        w[0] = in1_addr;
        w[2] = out_addr;
        w[4] = idx_addr;
    }
}

}  // namespace ttnn::prim
