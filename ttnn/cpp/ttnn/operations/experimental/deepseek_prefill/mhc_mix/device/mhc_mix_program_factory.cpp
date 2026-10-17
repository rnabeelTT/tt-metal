// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_mix_program_factory.hpp"
#include "mhc_mix_device_operation_types.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

namespace {
constexpr uint32_t CB_X = tt::CBIndex::c_0;   // the unit's Kc x tiles (one row tile, one K chunk)
constexpr uint32_t CB_FN = tt::CBIndex::c_1;  // the chunk's Kc projection tiles
constexpr uint32_t CB_XH = tt::CBIndex::c_2;  // x split for the matmuls: hi (10 significant bits)
constexpr uint32_t CB_XL = tt::CBIndex::c_3;  // ... lo
constexpr uint32_t CB_OUT = tt::CBIndex::c_16;

void add_cb(ProgramDescriptor& desc, const CoreRangeSet& cores, uint32_t index, uint32_t n_tiles, uint32_t tile_size) {
    desc.cbs.push_back(CBDescriptor{
        .total_size = n_tiles * tile_size,
        .core_ranges = cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index),
            .data_format = tt::DataFormat::Float32,
            .page_size = tile_size}}},
    });
}
}  // namespace

// Work unit = (row tile r, K chunk c), numbered r * C + c; each core takes a contiguous run.
ProgramDescriptor MhcMixProgramFactory::create_descriptor(
    const MhcMixParams& a,
    const MhcMixTensorArgs& t,
    tensor_return_value_t& out,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    ProgramDescriptor desc;
    const uint32_t ts = tt::tile_size(tt::DataFormat::Float32);
    const uint32_t Tt = t.x.padded_shape()[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t Kt = t.x.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    const uint32_t C = a.num_chunks;
    const uint32_t Kc = Kt / C;

    const auto grid = t.x.device()->compute_with_storage_grid_size();
    const uint32_t num_cores_y = grid.y;
    auto [num_cores, all_cores, g1, g2, units_1, units_2] = split_work_to_cores(grid, Tt * C);

    add_cb(desc, all_cores, CB_X, Kc, ts);
    add_cb(desc, all_cores, CB_FN, Kc, ts);
    add_cb(desc, all_cores, CB_XH, Kc, ts);
    add_cb(desc, all_cores, CB_XL, Kc, ts);
    add_cb(desc, all_cores, CB_OUT, 4, ts);

    const std::string kdir = "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/mhc_mix/device/kernels/";
    auto* x_buf = t.x.buffer();
    auto* fn_buf = t.fn_t.buffer();
    auto* out_buf = out.buffer();

    std::vector<uint32_t> reader_ct = {Kt, Kc, C};
    TensorAccessorArgs(x_buf).append_to(reader_ct);
    TensorAccessorArgs(fn_buf).append_to(reader_ct);
    KernelDescriptor reader;
    reader.kernel_source = kdir + "dataflow/reader_mhc_mix.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = all_cores;
    reader.compile_time_args = std::move(reader_ct);
    reader.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_ct = {Tt, C};
    TensorAccessorArgs(out_buf).append_to(writer_ct);
    KernelDescriptor writer;
    writer.kernel_source = kdir + "dataflow/writer_mhc_mix.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = all_cores;
    writer.compile_time_args = std::move(writer_ct);
    writer.config = WriterConfigDescriptor{};

    KernelDescriptor compute;
    compute.kernel_source = kdir + "compute/mhc_mix_compute.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = all_cores;
    compute.compile_time_args = {Kc};
    // x is only copied into DST (unpacked straight to DST, full fp32) and split there into hi / lo matmul halves
    std::vector<UnpackToDestMode> unpack_to_dest(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_to_dest[CB_X] = UnpackToDestMode::UnpackToDestFp32;
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
        .unpack_to_dest_mode = std::move(unpack_to_dest),
    };

    for (uint32_t i = 0, start = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        const uint32_t units = g1.contains(core) ? units_1 : units_2;
        {
            KernelDescriptor::RTArgList rt;
            rt.push_back(x_buf);
            rt.push_back(fn_buf);
            rt.push_back(units);
            rt.push_back(start);
            reader.emplace_runtime_args(core, rt);
        }
        {
            KernelDescriptor::RTArgList rt;
            rt.push_back(out_buf);
            rt.push_back(units);
            rt.push_back(start);
            writer.emplace_runtime_args(core, rt);
        }
        {
            KernelDescriptor::RTArgList rt;
            rt.push_back(units);
            compute.emplace_runtime_args(core, rt);
        }
        start += units;
    }
    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::experimental::prim
