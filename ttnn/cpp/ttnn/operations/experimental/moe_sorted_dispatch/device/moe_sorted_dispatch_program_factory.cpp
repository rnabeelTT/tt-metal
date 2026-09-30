// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_sorted_dispatch_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/hal.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

MoeSortedDispatchProgramFactory::cached_program_t MoeSortedDispatchProgramFactory::create(
    const MoeSortedDispatchParams& a, const MoeSortedDispatchInputs& t, std::vector<Tensor>& out) {
    Program program{};
    const CoreCoord core{0, 0};
    const uint32_t R = a.num_tokens * a.top_k;
    const uint32_t align = hal::get_l1_alignment();

    // one L1 scratch buffer per operand (whole stick each; single-toucher, no FIFO use)
    auto make_cb = [&](uint32_t index, uint32_t bytes, tt::DataFormat fmt) {
        const uint32_t sz = tt::align(bytes, align);
        CreateCircularBuffer(program, core, CircularBufferConfig(sz, {{index, fmt}}).set_page_size(index, sz));
        return sz;
    };
    // inputs are the tiled [.., T, k] top-k outputs: ceil(T/32) tiles each (one tile column since k <= 32)
    const uint32_t num_tiles = (a.num_tokens + 31) / 32;
    const uint32_t ids_tile_bytes = static_cast<uint32_t>(t.ids.buffer()->page_size());
    const uint32_t sc_tile_bytes = static_cast<uint32_t>(t.scores.buffer()->page_size());
    const uint32_t ids_elem_bytes = t.ids.dtype() == DataType::UINT16 ? 2 : 4;
    const uint32_t ids_bytes = make_cb(tt::CBIndex::c_0, num_tiles * ids_tile_bytes, tt::DataFormat::UInt32);
    const uint32_t sc_bytes = make_cb(tt::CBIndex::c_1, num_tiles * sc_tile_bytes, tt::DataFormat::Float16_b);
    const uint32_t src_bytes = make_cb(tt::CBIndex::c_2, a.dispatched_rows * 4, tt::DataFormat::UInt32);
    const uint32_t w_bytes = make_cb(tt::CBIndex::c_3, a.dispatched_rows * 2, tt::DataFormat::Float16_b);
    const uint32_t gid_bytes = make_cb(tt::CBIndex::c_4, a.num_groups * 2, tt::DataFormat::UInt16);
    const uint32_t und_bytes = make_cb(tt::CBIndex::c_5, R * 4, tt::DataFormat::UInt32);
    make_cb(tt::CBIndex::c_6, a.num_experts * 4 * 3, tt::DataFormat::UInt32);  // counts / offsets / ranks

    std::vector<uint32_t> ct = {
        R,
        a.num_tokens,
        a.top_k,
        a.num_experts,
        a.group,
        a.dispatched_rows,
        a.num_groups,
        static_cast<uint32_t>(t.ids.buffer()->page_size()),
        static_cast<uint32_t>(t.scores.buffer()->page_size()),
        static_cast<uint32_t>(out[0].buffer()->page_size()),
        static_cast<uint32_t>(out[1].buffer()->page_size()),
        static_cast<uint32_t>(out[2].buffer()->page_size()),
        static_cast<uint32_t>(out[3].buffer()->page_size()),
        num_tiles,
        ids_elem_bytes,
        a.pad_group_id,
    };
    (void)ids_bytes;
    (void)sc_bytes;
    (void)src_bytes;
    (void)w_bytes;
    (void)gid_bytes;
    (void)und_bytes;
    TensorAccessorArgs(*t.ids.buffer()).append_to(ct);
    TensorAccessorArgs(*t.scores.buffer()).append_to(ct);
    for (int i = 0; i < 4; ++i) {
        TensorAccessorArgs(*out[i].buffer()).append_to(ct);
    }
    const std::unordered_map<std::string, uint32_t> named = {
        {"cb_ids", tt::CBIndex::c_0},
        {"cb_scores", tt::CBIndex::c_1},
        {"cb_src", tt::CBIndex::c_2},
        {"cb_w", tt::CBIndex::c_3},
        {"cb_gid", tt::CBIndex::c_4},
        {"cb_und", tt::CBIndex::c_5},
        {"cb_scratch", tt::CBIndex::c_6},
    };
    auto kernel = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe_sorted_dispatch/device/kernels/moe_sorted_dispatch.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = ct,
            .named_compile_args = named});
    SetRuntimeArgs(
        program,
        kernel,
        core,
        {t.ids.buffer()->address(),
         t.scores.buffer()->address(),
         out[0].buffer()->address(),
         out[1].buffer()->address(),
         out[2].buffer()->address(),
         out[3].buffer()->address()});
    return {std::move(program), shared_variables_t{kernel, core}};
}

void MoeSortedDispatchProgramFactory::override_runtime_arguments(
    cached_program_t& cached,
    const MoeSortedDispatchParams&,
    const MoeSortedDispatchInputs& t,
    std::vector<Tensor>& out) {
    auto& args = GetRuntimeArgs(cached.program, cached.shared_variables.kernel, cached.shared_variables.core);
    args[0] = t.ids.buffer()->address();
    args[1] = t.scores.buffer()->address();
    for (int i = 0; i < 4; ++i) {
        args[2 + i] = out[i].buffer()->address();
    }
}

}  // namespace ttnn::experimental::prim
