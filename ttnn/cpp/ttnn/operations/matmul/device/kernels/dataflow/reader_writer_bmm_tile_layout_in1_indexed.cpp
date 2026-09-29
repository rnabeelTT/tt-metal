// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Group-parallel ("grouped") sparse matmul, in1 reader + output writer.
//
// This core owns `batch` consecutive sparse groups starting at `start_batch` (host-static). For every
// group it reads the WHOLE [K, N] weight block of expert `indices[start_batch + b]` (the ids are
// device-resident: they come from the routing of this very forward pass, so they are read from the
// `indices` tensor at run time, never baked into runtime args, which keeps the program trace-safe) and
// writes the group's compact output block [M, N] at slot `start_batch + b`. The activations of the
// group are read by the companion in0 reader (reader_bmm_tile_layout_in0.cpp), which walks the
// compact group axis with the same start.
//
// Derived from reader_writer_bmm_tile_layout_in1.cpp (interleaved in1 / out only; no sharded variants,
// no fused bias). Compile-time args 0..18 are identical to that kernel so the same host-side
// geometry code applies; the indices accessor args follow the in1 and out accessor args.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t rt_args_idx = 0;
    const uint32_t in1_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t batch = get_arg_val<uint32_t>(rt_args_idx++);  // groups owned by this core
    const uint32_t out_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t out_tensor_start_tile_id = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t idx_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t start_batch = get_arg_val<uint32_t>(rt_args_idx++);  // first compact group of this core

    constexpr uint32_t in1_tensor_stride_w = get_compile_time_arg_val(0);
    constexpr uint32_t in1_tensor_stride_h = get_compile_time_arg_val(1);
    constexpr uint32_t in1_tensor_next_block_stride = get_compile_time_arg_val(2);
    constexpr uint32_t in1_block_w = get_compile_time_arg_val(3);
    constexpr uint32_t in1_block_h = get_compile_time_arg_val(4);
    constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(6);
    // arg 7 (bcast_B) is unused: in1 is always indexed per group
    constexpr uint32_t KtNt = get_compile_time_arg_val(8);
    constexpr uint32_t out_tensor_stride_w = get_compile_time_arg_val(9);
    constexpr uint32_t out_tensor_stride_h = get_compile_time_arg_val(10);
    constexpr uint32_t out_tensor_next_subblock_stride_w = get_compile_time_arg_val(11);
    constexpr uint32_t out_tensor_next_subblock_stride_h = get_compile_time_arg_val(12);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(13);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(14);
    constexpr uint32_t out_subblock_tile_count = get_compile_time_arg_val(15);
    constexpr uint32_t out_num_subblocks_w = get_compile_time_arg_val(16);
    constexpr uint32_t out_num_subblocks_h = get_compile_time_arg_val(17);
    constexpr uint32_t MtNt = get_compile_time_arg_val(18);
    constexpr uint32_t idx_page_bytes = get_compile_time_arg_val(19);
    constexpr uint32_t num_groups_total = get_compile_time_arg_val(20);  // length of the id list (bounds ASSERT)
    constexpr uint32_t num_experts = get_compile_time_arg_val(21);       // sparse groups in in1 (bounds ASSERT)

    constexpr uint32_t dfb_id_in1 = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t dfb_id_out0 = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t dfb_id_idx = get_named_compile_time_arg_val("cb_idx");

    constexpr auto in1_args = TensorAccessorArgs<22>();
    constexpr auto out_args = TensorAccessorArgs<in1_args.next_compile_time_args_offset()>();
    constexpr auto idx_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    Noc noc;
    DataflowBuffer dfb_in1(dfb_id_in1);
    DataflowBuffer dfb_out(dfb_id_out0);
    DataflowBuffer dfb_idx(dfb_id_idx);

    const uint32_t in1_single_tile_size_bytes = dfb_in1.get_tile_size();
    const uint32_t in1_aligned_tile_size_bytes =
        (in1_single_tile_size_bytes + (DRAM_ALIGNMENT - 1)) & ~(DRAM_ALIGNMENT - 1);
    const auto s1 = TensorAccessor(in1_args, in1_tensor_addr);
    const uint32_t output_single_tile_size_bytes = dfb_out.get_tile_size();
    const auto s_out = TensorAccessor(out_args, out_tensor_addr);
    const auto s_idx = TensorAccessor(idx_args, idx_tensor_addr);

    // The whole id list is one ROW_MAJOR stick (validated on the host): one page-0 read.
    dfb_idx.reserve_back(1);
    const uint32_t idx_l1_addr = dfb_idx.get_write_ptr();
    noc.async_read(s_idx, dfb_idx, idx_page_bytes, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    const volatile tt_l1_ptr uint16_t* ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(idx_l1_addr);

    for (uint32_t b = 0; b < batch; ++b) {
        const uint32_t gb = start_batch + b;
        ASSERT(gb < num_groups_total);
        const uint32_t group_id = ids[gb];
        ASSERT(group_id < num_experts);
        uint32_t in1_tensor_current_block_start_tile_id = group_id * KtNt;
        for (uint32_t block = 0; block < num_blocks; ++block) {
            dfb_in1.reserve_back(in1_block_num_tiles);
            uint32_t in1_write_offset = 0;
            uint32_t in1_tensor_row_start_tile_id = in1_tensor_current_block_start_tile_id;
            for (uint32_t h = 0; h < in1_block_h; ++h) {
                uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
                for (uint32_t w = 0; w < in1_block_w; ++w) {
                    noc.async_read(
                        s1,
                        dfb_in1,
                        in1_single_tile_size_bytes,
                        {.page_id = in1_tensor_tile_id},
                        {.offset_bytes = in1_write_offset});
                    in1_write_offset += in1_aligned_tile_size_bytes;
                    in1_tensor_tile_id += in1_tensor_stride_w;
                }
                in1_tensor_row_start_tile_id += in1_tensor_stride_h;
            }
            in1_tensor_current_block_start_tile_id += in1_tensor_next_block_stride;
            noc.async_read_barrier();
            dfb_in1.push_back(in1_block_num_tiles);
        }

        uint32_t out_tensor_sbh_start_tile_id = out_tensor_start_tile_id;
        for (uint32_t sbh = 0; sbh < out_num_subblocks_h; ++sbh) {
            uint32_t out_tensor_sbw_start_tile_id = out_tensor_sbh_start_tile_id;
            for (uint32_t sbw = 0; sbw < out_num_subblocks_w; ++sbw) {
                uint32_t out_tensor_sb_row_start_tile_id = out_tensor_sbw_start_tile_id;
                dfb_out.wait_front(out_subblock_tile_count);
                uint32_t out_read_offset = 0;
                for (uint32_t h = 0; h < out_subblock_h; ++h) {
                    uint32_t out_tensor_tile_id = out_tensor_sb_row_start_tile_id;
                    for (uint32_t w = 0; w < out_subblock_w; ++w) {
                        noc.async_write(
                            dfb_out,
                            s_out,
                            output_single_tile_size_bytes,
                            {.offset_bytes = out_read_offset},
                            {.page_id = out_tensor_tile_id});
                        out_read_offset += output_single_tile_size_bytes;
                        out_tensor_tile_id += out_tensor_stride_w;
                    }
                    out_tensor_sb_row_start_tile_id += out_tensor_stride_h;
                }
                noc.async_write_barrier();
                dfb_out.pop_front(out_subblock_tile_count);
                out_tensor_sbw_start_tile_id += out_tensor_next_subblock_stride_w;
            }
            out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
        }
        out_tensor_start_tile_id += MtNt;
    }
    dfb_idx.push_back(1);
}
