// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// dsa_decode_scores: work unit = (slot s, 32-key block j), S * num_blocks units split over the grid (a core's run is
// contiguous, so q / weights of a slot are read once per run). Per active unit (j * 32 < limits[s]) the reader reads
// the block's D/32 key tiles through page_table[s], compute forms relu(q_s k^T) [32 heads x 32 keys] and the
// head-weighted row w_s . (.) [1 x 32], the writer writes that row into row s of the [S, num_blocks*32] output; an
// inactive unit is not read or computed and its row segment is written as -inf.

#include "dsa_decode_scores_program_factory.hpp"
#include "dsa_decode_scores_device_operation_types.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;
using tt::constants::TILE_WIDTH;

namespace {
constexpr uint32_t CB_Q = tt::CBIndex::c_0;       // q_s [32 heads, D]: D/32 bf16 tiles
constexpr uint32_t CB_W = tt::CBIndex::c_1;       // w_s row 0 (fp32 tile, rows 1..31 zero)
constexpr uint32_t CB_K = tt::CBIndex::c_2;       // one key block [32 keys, D]: D/32 bf16 tiles
constexpr uint32_t CB_CTRL = tt::CBIndex::c_3;    // reader -> compute: unit active flag
constexpr uint32_t CB_PT = tt::CBIndex::c_4;      // reader: page-table row of the current slot
constexpr uint32_t CB_LIM = tt::CBIndex::c_5;     // reader / writer: limits row
constexpr uint32_t CB_SC = tt::CBIndex::c_6;      // relu(q k^T) fp32 tile
constexpr uint32_t CB_NEGINF = tt::CBIndex::c_7;  // writer: one -inf face row
constexpr uint32_t CB_WLIM = tt::CBIndex::c_8;    // writer: its own limits row
constexpr uint32_t CB_OUT = tt::CBIndex::c_16;    // [1 x 32] score row (row 0 of an fp32 tile)
constexpr uint32_t CTRL_PAGE = 16;

void add_cb(
    ProgramDescriptor& d, const CoreRangeSet& cores, uint32_t idx, uint32_t pages, uint32_t page, tt::DataFormat f) {
    d.cbs.push_back(CBDescriptor{
        .total_size = pages * page,
        .core_ranges = cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(idx), .data_format = f, .page_size = page}}},
    });
}

uint32_t aligned(uint32_t bytes) {
    const uint32_t a = hal::get_dram_alignment();
    return ((bytes + a - 1) / a) * a;
}
}  // namespace

ProgramDescriptor DsaDecodeScoresProgramFactory::create_descriptor(
    const DsaDecodeScoresParams& a,
    const DsaDecodeScoresTensorArgs& t,
    tensor_return_value_t& out,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    ProgramDescriptor desc;
    const uint32_t S = t.weights.logical_shape()[-2];
    const uint32_t Dt = t.q.logical_shape()[-1] / TILE_WIDTH;
    const uint32_t nblk = a.num_blocks;
    const uint32_t bf16_tile = tt::tile_size(tt::DataFormat::Float16_b);
    const uint32_t fp32_tile = tt::tile_size(tt::DataFormat::Float32);
    const uint32_t pt_row_bytes = aligned(t.page_table.logical_shape()[-1] * 4);
    const uint32_t lim_bytes = aligned(t.limits.logical_volume() * 4);

    const auto grid = t.q.device()->compute_with_storage_grid_size();
    const uint32_t num_cores_y = grid.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, units_1, units_2] = split_work_to_cores(grid, S * nblk);

    add_cb(desc, all_cores, CB_Q, 2 * Dt, bf16_tile, tt::DataFormat::Float16_b);
    add_cb(desc, all_cores, CB_W, 2, fp32_tile, tt::DataFormat::Float32);
    add_cb(desc, all_cores, CB_K, 2 * Dt, bf16_tile, tt::DataFormat::Float16_b);
    add_cb(desc, all_cores, CB_CTRL, 2, CTRL_PAGE, tt::DataFormat::UInt32);
    add_cb(desc, all_cores, CB_PT, 1, pt_row_bytes, tt::DataFormat::UInt32);
    add_cb(desc, all_cores, CB_LIM, 1, lim_bytes, tt::DataFormat::UInt32);
    add_cb(desc, all_cores, CB_SC, 2, fp32_tile, tt::DataFormat::Float32);
    add_cb(desc, all_cores, CB_NEGINF, 1, 64, tt::DataFormat::Float32);
    add_cb(desc, all_cores, CB_WLIM, 1, lim_bytes, tt::DataFormat::UInt32);
    add_cb(desc, all_cores, CB_OUT, 2, fp32_tile, tt::DataFormat::Float32);

    const std::string kdir = "ttnn/cpp/ttnn/operations/experimental/dsa_decode_scores/device/kernels/";
    auto* q_buf = t.q.buffer();
    auto* w_buf = t.weights.buffer();
    auto* k_buf = t.k_pool.buffer();
    auto* pt_buf = t.page_table.buffer();
    auto* lim_buf = t.limits.buffer();
    auto* out_buf = out.buffer();

    std::vector<uint32_t> reader_ct = {Dt, nblk, pt_row_bytes, lim_bytes};
    TensorAccessorArgs(q_buf).append_to(reader_ct);
    TensorAccessorArgs(w_buf).append_to(reader_ct);
    TensorAccessorArgs(k_buf).append_to(reader_ct);
    TensorAccessorArgs(pt_buf).append_to(reader_ct);
    TensorAccessorArgs(lim_buf).append_to(reader_ct);
    KernelDescriptor reader;
    reader.kernel_source = kdir + "dataflow/reader_dsa_decode_scores.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = all_cores;
    reader.compile_time_args = std::move(reader_ct);
    reader.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_ct = {nblk, lim_bytes};
    TensorAccessorArgs(out_buf).append_to(writer_ct);
    TensorAccessorArgs(lim_buf).append_to(writer_ct);
    KernelDescriptor writer;
    writer.kernel_source = kdir + "dataflow/writer_dsa_decode_scores.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = all_cores;
    writer.compile_time_args = std::move(writer_ct);
    writer.config = WriterConfigDescriptor{};

    KernelDescriptor compute;
    compute.kernel_source = kdir + "compute/dsa_decode_scores.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = all_cores;
    compute.compile_time_args = {Dt, nblk};
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
        .math_approx_mode = false,
    };

    for (uint32_t i = 0, start = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        const uint32_t units = core_group_1.contains(core) ? units_1 : units_2;
        {
            KernelDescriptor::RTArgList rt;
            for (auto* b : {q_buf, w_buf, k_buf, pt_buf, lim_buf}) {
                rt.push_back(b);
            }
            rt.push_back(start);
            rt.push_back(units);
            reader.emplace_runtime_args(core, rt);
        }
        {
            KernelDescriptor::RTArgList rt;
            rt.push_back(out_buf);
            rt.push_back(lim_buf);
            rt.push_back(start);
            rt.push_back(units);
            writer.emplace_runtime_args(core, rt);
        }
        {
            KernelDescriptor::RTArgList rt;
            rt.push_back(start);
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
