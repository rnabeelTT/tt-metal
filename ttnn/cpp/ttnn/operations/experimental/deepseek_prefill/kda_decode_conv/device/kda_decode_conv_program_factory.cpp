// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "kda_decode_conv_program_factory.hpp"
#include "kda_decode_conv_device_operation_types.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

namespace {
constexpr uint32_t CB_CUR = tt::CBIndex::c_0;    // current row tile (bf16, from proj)
constexpr uint32_t CB_H = tt::CBIndex::c_1;      // h0, h1, h2 tiles (bf16)
constexpr uint32_t CB_TAP = tt::CBIndex::c_2;    // tap0..tap3 tiles (fp32, row 0 broadcast down by the reader)
constexpr uint32_t CB_ACT = tt::CBIndex::c_3;    // slot mask (fp32, column 0 broadcast across by the reader)
constexpr uint32_t CB_QKV = tt::CBIndex::c_16;   // qkv out (fp32)
constexpr uint32_t CB_HOUT = tt::CBIndex::c_17;  // shifted history out (bf16)

void add_cb(ProgramDescriptor& desc, const CoreRangeSet& cores, uint32_t index, uint32_t n, tt::DataFormat df) {
    const uint32_t ts = tt::tile_size(df);
    desc.cbs.push_back(CBDescriptor{
        .total_size = n * ts,
        .core_ranges = cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index), .data_format = df, .page_size = ts}}},
    });
}
}  // namespace

ProgramDescriptor KdaDecodeConvProgramFactory::create_descriptor(
    const KdaDecodeConvParams& a,
    const KdaDecodeConvTensorArgs& t,
    tensor_return_value_t& out,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    ProgramDescriptor desc;
    const auto bf16 = tt::DataFormat::Float16_b;
    const auto fp32 = tt::DataFormat::Float32;
    const uint32_t Wt = a.width / tt::constants::TILE_WIDTH;
    const uint32_t rows = t.proj.logical_shape()[2];

    auto grid = t.proj.device()->compute_with_storage_grid_size();
    const uint32_t num_cores_y = grid.y;
    auto [num_cores, all_cores, g1, g2, n1, n2] = split_work_to_cores(grid, Wt);

    add_cb(desc, all_cores, CB_CUR, 2, bf16);
    add_cb(desc, all_cores, CB_H, 6, bf16);
    add_cb(desc, all_cores, CB_TAP, 8, fp32);
    add_cb(desc, all_cores, CB_ACT, 1, fp32);
    add_cb(desc, all_cores, CB_QKV, 2, fp32);
    add_cb(desc, all_cores, CB_HOUT, 6, bf16);

    Buffer* bufs[] = {
        t.proj.buffer(),
        t.h0.buffer(),
        t.h1.buffer(),
        t.h2.buffer(),
        t.tap0.buffer(),
        t.tap1.buffer(),
        t.tap2.buffer(),
        t.tap3.buffer(),
        t.active.buffer()};
    std::vector<uint32_t> reader_ct = {CB_CUR, CB_H, CB_TAP, CB_ACT};
    for (Buffer* b : bufs) {
        TensorAccessorArgs(b).append_to(reader_ct);
    }
    const std::string kdir = "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/kda_decode_conv/device/kernels/";
    KernelDescriptor reader;
    reader.kernel_source = kdir + "dataflow/reader_kda_decode_conv.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = all_cores;
    reader.compile_time_args = std::move(reader_ct);
    reader.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_ct = {CB_QKV, CB_HOUT};
    TensorAccessorArgs(out.buffer()).append_to(writer_ct);
    TensorAccessorArgs(t.h0.buffer()).append_to(writer_ct);
    TensorAccessorArgs(t.h1.buffer()).append_to(writer_ct);
    TensorAccessorArgs(t.h2.buffer()).append_to(writer_ct);
    KernelDescriptor writer;
    writer.kernel_source = kdir + "dataflow/writer_kda_decode_conv.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = all_cores;
    writer.compile_time_args = std::move(writer_ct);
    writer.config = WriterConfigDescriptor{};

    ComputeConfigDescriptor::UnpackToDestModes modes(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    modes[CB_TAP] = UnpackToDestMode::UnpackToDestFp32;
    modes[CB_ACT] = UnpackToDestMode::UnpackToDestFp32;
    KernelDescriptor compute;
    compute.kernel_source = kdir + "compute/kda_decode_conv_compute.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = all_cores;
    compute.compile_time_args = {CB_CUR, CB_H, CB_TAP, CB_ACT, CB_QKV, CB_HOUT};
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
        .unpack_to_dest_mode = modes,
        .math_approx_mode = false};

    for (uint32_t i = 0, start = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        const uint32_t n = g1.contains(core) ? n1 : n2;
        KernelDescriptor::RTArgList rr;
        for (Buffer* b : bufs) {
            rr.push_back(b);
        }
        rr.push_back(start);
        rr.push_back(n);
        rr.push_back(rows);
        reader.emplace_runtime_args(core, rr);
        KernelDescriptor::RTArgList rw;
        rw.push_back(out.buffer());
        rw.push_back(t.h0.buffer());
        rw.push_back(t.h1.buffer());
        rw.push_back(t.h2.buffer());
        rw.push_back(start);
        rw.push_back(n);
        writer.emplace_runtime_args(core, rw);
        KernelDescriptor::RTArgList rc;
        rc.push_back(n);
        compute.emplace_runtime_args(core, rc);
        start += n;
    }
    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::experimental::prim
