// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// kda.recurrent_decode: the B*H (user, head) recurrences are split over the compute grid, a contiguous run of heads per
// core. Per head the reader gathers the head's rows of q, k, v, decay (two 64-byte face-row reads per tile, straight
// out of the packed [1,1,R,C] projections) and beta (one element), plus the head's fp32 state [K,V]; the compute kernel
// runs the step (derived from ttnn.transformer.fused_recurrent_gated_delta_rule, with a per-key-channel decay and the
// q / k L2 norms in-kernel); the writer writes the state back in place and the output row into [1,1,B*H,V].

#include "recurrent_decode_program_factory.hpp"
#include "recurrent_decode_device_operation_types.hpp"

#include <bit>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;
using tt::constants::TILE_WIDTH;

namespace {
// CB plan (all fp32). Kept in sync with the reader / compute / writer kernels. "X" = UnpackToDestFp32: the compute
// kernel only copies / transposes those into DST (full fp32); the others feed the FPU (matmul) and hold the hi / lo
// halves of split fp32 operands (hi sized so the multiplier uses all of it).
constexpr uint32_t CB_Q = tt::CBIndex::c_0;         // X [1,K] head row (rows 1..31 zero)
constexpr uint32_t CB_K = tt::CBIndex::c_1;         // X [1,K]
constexpr uint32_t CB_V = tt::CBIndex::c_2;         // X [1,V]
constexpr uint32_t CB_DECAY = tt::CBIndex::c_3;     // X [1,K]
constexpr uint32_t CB_BETA = tt::CBIndex::c_4;      // X [0,0] = beta, rest zero
constexpr uint32_t CB_S = tt::CBIndex::c_5;         // X [K,V] state in
constexpr uint32_t CB_OUT = tt::CBIndex::c_6;       // [1,V] o row
constexpr uint32_t CB_ST = tt::CBIndex::c_7;        // [K,V] state out (writer)
constexpr uint32_t CB_SDH = tt::CBIndex::c_8;       // [K,V] diag(decay) S, hi
constexpr uint32_t CB_SDL = tt::CBIndex::c_9;       // [K,V] diag(decay) S, lo
constexpr uint32_t CB_VREAD = tt::CBIndex::c_10;    // X [1,V]
constexpr uint32_t CB_UH = tt::CBIndex::c_11;       // [1,V] u hi
constexpr uint32_t CB_KCOL = tt::CBIndex::c_12;     // [K,1] kn hi
constexpr uint32_t CB_DCOL = tt::CBIndex::c_13;     // X [K,1]
constexpr uint32_t CB_QNH = tt::CBIndex::c_14;      // [1,K]
constexpr uint32_t CB_KNH = tt::CBIndex::c_15;      // [1,K]
constexpr uint32_t CB_NRM = tt::CBIndex::c_16;      // X [0,0] = 1 / ||x|| (scaled)
constexpr uint32_t CB_SCRATCH = tt::CBIndex::c_17;  // reader: one 64-byte face row (beta)
constexpr uint32_t CB_XH = tt::CBIndex::c_18;       // [1,K] q or k, hi (norm)
constexpr uint32_t CB_XL = tt::CBIndex::c_19;       // [1,K] q or k, lo (norm)
constexpr uint32_t CB_QNL = tt::CBIndex::c_20;      // [1,K]
constexpr uint32_t CB_KNL = tt::CBIndex::c_21;      // [1,K]
constexpr uint32_t CB_KCOLL = tt::CBIndex::c_22;    // [K,1] kn lo
constexpr uint32_t CB_UL = tt::CBIndex::c_23;       // [1,V] u lo
constexpr uint32_t CB_SDX = tt::CBIndex::c_24;      // X [K,V] diag(decay) S, exact (rank-1 seed)
constexpr uint32_t CB_QSD = tt::CBIndex::c_25;      // X [1,V] qn . sd
constexpr uint32_t CB_QK = tt::CBIndex::c_26;       // X [0,0] = qn . kn
constexpr uint32_t CB_KN1H = tt::CBIndex::c_27;     // [1,K] kn split for the matmul-in1 role, hi
constexpr uint32_t CB_KN1L = tt::CBIndex::c_28;     // ... lo
constexpr uint32_t UNPACK_TO_DEST_CBS[] = {
    CB_Q, CB_K, CB_V, CB_DECAY, CB_BETA, CB_S, CB_VREAD, CB_DCOL, CB_NRM, CB_SDX, CB_QSD, CB_QK};

void add_cb(ProgramDescriptor& desc, const CoreRangeSet& cores, uint32_t index, uint32_t n_tiles, uint32_t tile_size) {
    desc.cbs.push_back(CBDescriptor{
        .total_size = n_tiles * tile_size,
        .core_ranges = cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index),
            .data_format = tt::DataFormat::Float32,
            .page_size = tile_size,
        }}},
    });
}
}  // namespace

ProgramDescriptor KdaRecurrentDecodeProgramFactory::create_descriptor(
    const KdaRecurrentDecodeParams& a,
    const KdaRecurrentDecodeTensorArgs& t,
    tensor_return_value_t& out,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    ProgramDescriptor desc;
    const uint32_t ts = tt::tile_size(tt::DataFormat::Float32);
    const uint32_t H = a.num_heads;
    const uint32_t Kt = a.key_dim / TILE_WIDTH;
    const uint32_t Vt = a.value_dim / TILE_WIDTH;
    const uint32_t kv = Kt * Vt;
    const uint32_t N = a.batch * H;

    const uint32_t qkv_ct = t.qkv.padded_shape()[-1] / TILE_WIDTH;
    const uint32_t decay_ct = t.decay.padded_shape()[-1] / TILE_WIDTH;
    const uint32_t beta_ct = t.beta.padded_shape()[-1] / TILE_WIDTH;

    const auto grid = t.state.device()->compute_with_storage_grid_size();
    const uint32_t num_cores_y = grid.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, heads_1, heads_2] = split_work_to_cores(grid, N);

    // inputs double-buffered (the next head's reads overlap this head's compute)
    add_cb(desc, all_cores, CB_Q, 2 * Kt, ts);
    add_cb(desc, all_cores, CB_K, 2 * Kt, ts);
    add_cb(desc, all_cores, CB_V, 2 * Vt, ts);
    add_cb(desc, all_cores, CB_DECAY, 2 * Kt, ts);
    add_cb(desc, all_cores, CB_BETA, 2, ts);
    add_cb(desc, all_cores, CB_S, 2 * kv, ts);
    add_cb(desc, all_cores, CB_OUT, 2 * Vt, ts);
    add_cb(desc, all_cores, CB_ST, 2 * kv, ts);
    add_cb(desc, all_cores, CB_SDH, kv, ts);
    add_cb(desc, all_cores, CB_SDL, kv, ts);
    add_cb(desc, all_cores, CB_SDX, kv, ts);
    add_cb(desc, all_cores, CB_VREAD, Vt, ts);
    add_cb(desc, all_cores, CB_QSD, Vt, ts);
    add_cb(desc, all_cores, CB_QK, 1, ts);
    add_cb(desc, all_cores, CB_UH, Vt, ts);
    add_cb(desc, all_cores, CB_UL, Vt, ts);
    add_cb(desc, all_cores, CB_KCOL, Kt, ts);
    add_cb(desc, all_cores, CB_KCOLL, Kt, ts);
    add_cb(desc, all_cores, CB_DCOL, Kt, ts);
    add_cb(desc, all_cores, CB_QNH, Kt, ts);
    add_cb(desc, all_cores, CB_QNL, Kt, ts);
    add_cb(desc, all_cores, CB_KNH, Kt, ts);
    add_cb(desc, all_cores, CB_KNL, Kt, ts);
    add_cb(desc, all_cores, CB_KN1H, Kt, ts);
    add_cb(desc, all_cores, CB_KN1L, Kt, ts);
    add_cb(desc, all_cores, CB_XH, Kt, ts);
    add_cb(desc, all_cores, CB_XL, Kt, ts);
    add_cb(desc, all_cores, CB_NRM, 1, ts);
    add_cb(desc, all_cores, CB_SCRATCH, 1, ts);

    const std::string kdir = "ttnn/cpp/ttnn/operations/experimental/kda/recurrent_decode/device/kernels/";
    auto* state_buf = t.state.buffer();
    auto* qkv_buf = t.qkv.buffer();
    auto* decay_buf = t.decay.buffer();
    auto* beta_buf = t.beta.buffer();
    auto* out_buf = out.buffer();

    std::vector<uint32_t> reader_ct = {Kt, Vt, H, qkv_ct, decay_ct, beta_ct};
    TensorAccessorArgs(state_buf).append_to(reader_ct);
    TensorAccessorArgs(qkv_buf).append_to(reader_ct);
    TensorAccessorArgs(decay_buf).append_to(reader_ct);
    TensorAccessorArgs(beta_buf).append_to(reader_ct);
    KernelDescriptor reader;
    reader.kernel_source = kdir + "dataflow/reader_recurrent_decode.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = all_cores;
    reader.compile_time_args = std::move(reader_ct);
    reader.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_ct = {Kt, Vt};
    TensorAccessorArgs(state_buf).append_to(writer_ct);
    TensorAccessorArgs(out_buf).append_to(writer_ct);
    KernelDescriptor writer;
    writer.kernel_source = kdir + "dataflow/writer_recurrent_decode.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = all_cores;
    writer.compile_time_args = std::move(writer_ct);
    writer.config = WriterConfigDescriptor{};

    KernelDescriptor compute;
    compute.kernel_source = kdir + "compute/recurrent_decode.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = all_cores;
    compute.compile_time_args = {Kt, Vt, std::bit_cast<uint32_t>(a.eps), std::bit_cast<uint32_t>(a.q_scale)};
    // fp32 operands that the kernel only moves into DST are unpacked straight to DST (no TF32 rounding in srcA)
    std::vector<UnpackToDestMode> unpack_to_dest(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (uint32_t cb : UNPACK_TO_DEST_CBS) {
        unpack_to_dest[cb] = UnpackToDestMode::UnpackToDestFp32;
    }
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
        .unpack_to_dest_mode = std::move(unpack_to_dest),
        .math_approx_mode = false,
    };

    for (uint32_t i = 0, start = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        const uint32_t heads = core_group_1.contains(core) ? heads_1 : heads_2;
        {
            KernelDescriptor::RTArgList rt;
            rt.push_back(state_buf);
            rt.push_back(qkv_buf);
            rt.push_back(decay_buf);
            rt.push_back(beta_buf);
            rt.push_back(start);
            rt.push_back(heads);
            reader.emplace_runtime_args(core, rt);
        }
        {
            KernelDescriptor::RTArgList rt;
            rt.push_back(state_buf);
            rt.push_back(out_buf);
            rt.push_back(start);
            rt.push_back(heads);
            writer.emplace_runtime_args(core, rt);
        }
        {
            KernelDescriptor::RTArgList rt;
            rt.push_back(heads);
            compute.emplace_runtime_args(core, rt);
        }
        start += heads;
    }

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::experimental::prim
