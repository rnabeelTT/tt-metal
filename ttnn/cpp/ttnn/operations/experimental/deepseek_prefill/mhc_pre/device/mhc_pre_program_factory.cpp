// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_pre_program_factory.hpp"
#include "mhc_pre_device_operation_types.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

namespace {
constexpr uint32_t CB_IN = tt::CBIndex::c_0;      // the n residual-stream tiles of one (token-tile, column-tile)
constexpr uint32_t CB_PRE = tt::CBIndex::c_1;     // raw pre tile of the token-tile
constexpr uint32_t CB_CONSTS = tt::CBIndex::c_2;  // n column-broadcast tiles, resident
constexpr uint32_t CB_COEF = tt::CBIndex::c_3;    // n broadcast pre_i tiles of the token-tile
constexpr uint32_t CB_PH = tt::CBIndex::c_4;      // pre split for the expansion matmul: hi (TF32-exact)
constexpr uint32_t CB_PL = tt::CBIndex::c_5;      // ... lo
constexpr uint32_t CB_OUT = tt::CBIndex::c_16;

void add_cb(
    ProgramDescriptor& desc,
    const CoreRangeSet& cores,
    uint32_t index,
    uint32_t n_tiles,
    uint32_t tile_size,
    tt::DataFormat df) {
    desc.cbs.push_back(CBDescriptor{
        .total_size = n_tiles * tile_size,
        .core_ranges = cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index),
            .data_format = df,
            .page_size = tile_size,
        }}},
    });
}
}  // namespace

ProgramDescriptor MhcPreProgramFactory::create_descriptor(
    const MhcPreParams& operation_attributes,
    const MhcPreTensorArgs& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    ProgramDescriptor desc;
    const tt::DataFormat df = datatype_to_dataformat_converter(tensor_args.residual.dtype());
    const tt::DataFormat out_df = datatype_to_dataformat_converter(operation_attributes.output_dtype);
    const uint32_t tile_size = tt::tile_size(df);
    const uint32_t out_tile_size = tt::tile_size(out_df);
    const uint32_t n = operation_attributes.n;

    auto* residual_buffer = tensor_args.residual.buffer();
    auto* pre_buffer = tensor_args.pre.buffer();
    auto* consts_buffer = tensor_args.consts.buffer();
    auto* out_buffer = tensor_return_value.buffer();

    const auto& rps = tensor_args.residual.padded_shape();
    const uint32_t token_tiles = rps[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t col_tiles = rps[-1] / tt::constants::TILE_WIDTH / n;

    // One work unit = one (token-tile, column-tile): reads the n stream tiles at that position, writes one tile. Units
    // are numbered row-major (t0 * col_tiles + c0) and each core takes a contiguous run, so a core normally expands one
    // token-tile's coefficients once.
    const uint32_t num_units = token_tiles * col_tiles;
    const auto grid = tensor_args.residual.device()->compute_with_storage_grid_size();
    const uint32_t num_cores_y = grid.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_1, units_per_core_2] =
        split_work_to_cores(grid, num_units);

    add_cb(desc, all_cores, CB_IN, 2 * n, tile_size, df);
    add_cb(desc, all_cores, CB_PRE, 2, tile_size, df);
    add_cb(desc, all_cores, CB_CONSTS, n, tile_size, df);
    add_cb(desc, all_cores, CB_COEF, n, tile_size, df);
    add_cb(desc, all_cores, CB_PH, 1, tile_size, df);
    add_cb(desc, all_cores, CB_PL, 1, tile_size, df);
    add_cb(desc, all_cores, CB_OUT, 2, out_tile_size, out_df);

    const std::string kdir = "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/mhc_pre/device/kernels/";

    std::vector<uint32_t> reader_ct = {CB_IN, CB_PRE, CB_CONSTS, n, col_tiles};
    TensorAccessorArgs(residual_buffer).append_to(reader_ct);
    TensorAccessorArgs(pre_buffer).append_to(reader_ct);
    TensorAccessorArgs(consts_buffer).append_to(reader_ct);
    KernelDescriptor reader;
    reader.kernel_source = kdir + "dataflow/reader_mhc_pre.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = all_cores;
    reader.compile_time_args = std::move(reader_ct);
    reader.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_ct = {CB_OUT, col_tiles};
    TensorAccessorArgs(out_buffer).append_to(writer_ct);
    KernelDescriptor writer;
    writer.kernel_source = kdir + "dataflow/writer_mhc_pre.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = all_cores;
    writer.compile_time_args = std::move(writer_ct);
    writer.config = WriterConfigDescriptor{};

    KernelDescriptor compute;
    compute.kernel_source = kdir + "compute/mhc_pre_compute.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = all_cores;
    compute.compile_time_args = {n, col_tiles};
    // The FPU reads fp32 at TF32: for an fp32 op the kernel only copies the streams, pre and the expanded coefficients
    // into DST (unpacked straight to DST, full precision) and accumulates on the SFPU.
    std::vector<UnpackToDestMode> unpack_to_dest(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (df == tt::DataFormat::Float32) {
        for (uint32_t cb : {CB_IN, CB_PRE, CB_COEF}) {
            unpack_to_dest[cb] = UnpackToDestMode::UnpackToDestFp32;
        }
    }
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
        .unpack_to_dest_mode = std::move(unpack_to_dest),
    };

    for (uint32_t i = 0, start_unit = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t units = core_group_1.contains(core) ? units_per_core_1 : units_per_core_2;
        {
            KernelDescriptor::RTArgList rt;
            rt.push_back(residual_buffer);
            rt.push_back(pre_buffer);
            rt.push_back(consts_buffer);
            rt.push_back(units);
            rt.push_back(start_unit);
            reader.emplace_runtime_args(core, rt);
        }
        {
            KernelDescriptor::RTArgList rt;
            rt.push_back(out_buffer);
            rt.push_back(units);
            rt.push_back(start_unit);
            writer.emplace_runtime_args(core, rt);
        }
        {
            KernelDescriptor::RTArgList rt;
            rt.push_back(units);
            rt.push_back(start_unit);
            compute.emplace_runtime_args(core, rt);
        }
        start_unit += units;
    }

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::experimental::prim
