// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// sparse_sdpa writer: per token, drain the untilized [H, V_DIM] row-major block from compute (Sqt*vDHt
// tile-sized pages, row-major bytes inside) and write each head-row to the ROW-MAJOR [1,H,S,V_DIM] output.
// Also builds the three persistent compute-input tiles once at startup (the reader is busier): the reduce
// identity scaler, the col-identity (row-sum finalize), and the all-(-inf) mask tile. Optional per-head
// attention sinks are gathered once here too, while the reader loads Q and indices.
// Uses the object-based NoC + circular-buffer APIs.

#include <stdint.h>
#include "api/tile_constants.h"  // tt::constants::TILE_WIDTH
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp"  // generate_bcast_col_scalar
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "sparse_sdpa_gather.hpp"  // dual-NoC trid-ring gather helper (shared with the reader)

constexpr uint32_t one_bf16_packed = 0x3F803F80u;  // bf16(1.0) double-packed; generate_bcast_col_scalar uses >>16
constexpr uint16_t neg_inf_bf16 = 0xFF80u;

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::H);
    constexpr uint32_t S = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::S);
    constexpr uint32_t vDHt = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::V_DHT);
    // CB ids — passed by the factory (the SparseCB enum is the single source); order must match the
    // factory's writer CB-arg block (compile args 3..6).
    constexpr uint32_t cb_out_rm = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::CB_OUT_RM);
    constexpr uint32_t cb_scale = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::CB_SCALE);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::CB_COL_IDENTITY);
    constexpr uint32_t cb_neginf = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::CB_NEGINF);
    constexpr uint32_t out_elem_bytes =
        get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::OUT_ELEM_BYTES);  // output tensor dtype
    constexpr bool block_cyclic = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::BLOCK_CYCLIC) != 0;
    constexpr uint32_t bc_chunk_local = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::BC_CHUNK_LOCAL);
    constexpr uint32_t bc_sp = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::BC_SP);
    constexpr uint32_t bc_shard_stride_gap = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::BC_SHARD_STRIDE_GAP);
    constexpr uint32_t bc_slab_stride_gap = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::BC_SLAB_STRIDE_GAP);
    constexpr bool scaled_kv = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::SCALED_KV) != 0;
    constexpr uint32_t k_dim = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::K_DIM);
    constexpr uint32_t kv_elem_bytes = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::KV_ELEM_BYTES);
    constexpr uint32_t cb_idx = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::CB_IDX);
    constexpr uint32_t cb_kreq = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::CB_KREQ);
    constexpr uint32_t cb_kack = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::CB_KACK);
    constexpr uint32_t packed_row_bytes = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::PACKED_ROW_BYTES);
    constexpr bool use_attention_sink = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::USE_ATTENTION_SINK) != 0;
    constexpr uint32_t cb_attention_sink = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::CB_ATTENTION_SINK);
    constexpr uint32_t cb_sink_scratch = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::CB_SINK_SCRATCH);
    // Key split (1 = off): runtime arg 2 = this core's index j in its token group, then the NoC x/y of the group's
    // KV_SPLIT cores (participant t at args 3 + 2t, 4 + 2t).
    constexpr uint32_t kv_split = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::KV_SPLIT);
    constexpr bool kv_split_on = kv_split > 1;
    constexpr uint32_t cb_split_ctl = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::CB_SPLIT_CTL);
    constexpr uint32_t cb_split_done = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::CB_SPLIT_DONE);
    constexpr uint32_t cb_lmax = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::CB_LMAX);
    constexpr uint32_t cb_pmax = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::CB_PMAX);
    constexpr uint32_t cb_xo = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::CB_XO);
    constexpr uint32_t cb_xs = get_compile_time_arg_val(sparse_sdpa::writer_ct_arg::CB_XS);
    constexpr auto out_args =
        TensorAccessorArgs<sparse_sdpa::writer_ct_arg::END, sparse_sdpa::writer_common_arg::END>();
    // kv carries a RUNTIME tensor shape (T dim in common runtime args), so its accessor spans both arg streams.
    constexpr auto kv_args =
        TensorAccessorArgs<out_args.next_compile_time_args_offset(), out_args.next_common_runtime_args_offset()>();

    const uint32_t out_addr = get_common_arg_val<uint32_t>(sparse_sdpa::writer_common_arg::OUTPUT_ADDRESS);
    const uint32_t tok_start = get_arg_val<uint32_t>(0);
    const uint32_t tok_count = get_arg_val<uint32_t>(1);
    const uint32_t kv_addr =
        get_common_arg_val<uint32_t>(sparse_sdpa::writer_common_arg::KV_ADDRESS);  // dual-NoC K-half gather
    // Indexed KV cache: page offset (cache_batch_idx * T) selecting the cache's batch slot; 0 if not indexed.
    const uint32_t kv_batch_page_offset =
        get_common_arg_val<uint32_t>(sparse_sdpa::writer_common_arg::KV_BATCH_PAGE_OFFSET);

    constexpr uint32_t row_bytes = vDHt * tt::constants::TILE_WIDTH * out_elem_bytes;  // V_DIM bytes per head-row
    constexpr uint32_t block_tiles = (H / tt::constants::TILE_HEIGHT) * vDHt;          // Sqt*vDHt: the [H,V_DIM] block

    Noc noc;
    experimental::CB out_cb(cb_out_rm);
    const auto out = TensorAccessor(out_args, out_addr);
    // Dual-NoC K gather: the writer co-gathers half of each K chunk on ITS NoC into the shared cb_k_rm L1
    // (the reader owns the reserve/push; we fill rows [0,half) at the same write ptr). This spreads the
    // K response-data load over both NoC links. Indices live in the reader's shared cb_idx
    // scratch; the kv accessor + row size mirror the reader.
    constexpr uint32_t k_row_bytes = k_dim * kv_elem_bytes;
    constexpr uint32_t latent_row_bytes = vDHt * tt::constants::TILE_WIDTH;
    constexpr uint32_t scale_blocks = sparse_sdpa::scale_block_count(latent_row_bytes);
    const auto kv = TensorAccessor(kv_args, kv_addr);
    experimental::CB idx_cb(cb_idx), kreq_cb(cb_kreq), kack_cb(cb_kack);
    volatile tt_l1_ptr uint32_t* idx_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_cb.get_write_ptr());

    // --- persistent compute-input tiles (built once; the reader is busier with the K gather) ---
    // Reduce identity scaler (value 1.0; the softmax scale is applied in compute's exp).
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_scale,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        /*reduce_factor=*/1>();

    // Col-identity (column 0 = 1.0): compute matmul-reduces the partial row-sum against it to finalize the
    // within-tile reduction in normalize_row_streaming.
    generate_bcast_col_scalar(CircularBuffer(cb_col_identity), one_bf16_packed);

    // All-(-inf) tile (row 0; compute bcast-adds it to fully-masked key tiles). Only row 0 matters for the
    // row-broadcast add: the left and right half-rows live in face 0 and face 1 respectively.
    {
        experimental::CB neginf_cb(cb_neginf);
        neginf_cb.reserve_back(1);
        volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(neginf_cb.get_write_ptr());
        for (uint32_t c = 0; c < tt::constants::FACE_WIDTH; ++c) {
            p[c] = neg_inf_bf16;
            p[tt::constants::FACE_HW + c] = neg_inf_bf16;
        }
        neginf_cb.push_back(1);
    }

    if constexpr (use_attention_sink) {
        if (tok_count > 0) {
            constexpr auto sink_args = TensorAccessorArgs<
                kv_args.next_compile_time_args_offset(),
                kv_args.next_common_runtime_args_offset()>();
            const auto sink = TensorAccessor(sink_args, get_common_arg_val<uint32_t>(sparse_sdpa::writer_common_arg::ATTENTION_SINK_ADDRESS));
            experimental::CB sink_cb(cb_attention_sink), scratch(cb_sink_scratch);
            constexpr uint32_t sink_bytes = H * sizeof(uint16_t);
            constexpr uint32_t tile_bytes = tt::constants::TILE_HW * sizeof(uint16_t);
            constexpr uint32_t faces_per_row = tt::constants::TILE_WIDTH / tt::constants::FACE_WIDTH;
            static_assert(H % tt::constants::TILE_HEIGHT == 0);
            static_assert(sink_bytes % NOC_DRAM_READ_ALIGNMENT_BYTES == 0);
            constexpr uint32_t sink_tiles = H / tt::constants::TILE_HEIGHT;
            sink_cb.reserve_back(sink_tiles);
            scratch.reserve_back(1);
            noc.async_write_zeros(sink_cb, sink_tiles * tile_bytes);
            // [1,1,1,H] BF16 stores all heads in one row. Read once, then pack the scalars
            // into the first column of the compute tiles (one head per row).
            noc.async_read(sink, scratch, sink_bytes, {.page_id = 0}, {.offset_bytes = 0});
            noc.async_read_barrier();
            noc.write_zeros_l1_barrier();
            auto* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sink_cb.get_write_ptr());
            auto* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch.get_write_ptr());
            for (uint32_t h = 0; h < H; ++h) {
                const uint32_t row = h % tt::constants::TILE_HEIGHT;
                const uint32_t offset = (h / tt::constants::TILE_HEIGHT) * tt::constants::TILE_HW +
                                        (row / tt::constants::FACE_HEIGHT) * faces_per_row * tt::constants::FACE_HW +
                                        (row % tt::constants::FACE_HEIGHT) * tt::constants::FACE_WIDTH;
                dst[offset] = src[h];
            }
            sink_cb.push_back(sink_tiles);  // reused by every token on this core
        }
    }

    // Key split: the receive slots (cb_pmax slot j, cb_xo / cb_xs slot 0) sit at the same L1 offsets on every core,
    // and this core's final chain state (slot LOCAL_CHUNKS of cb_xo / cb_xs) is read straight from L1 (compute
    // holds those CBs' write pointer at their base for the token).
    uint32_t split_j = 0;
    constexpr uint32_t state_tile_bytes = tt::constants::TILE_HW * sizeof(uint16_t);  // bf16 flash state tiles
    uint32_t pmax_l1 = 0, xo_l1 = 0, xs_l1 = 0;
    if constexpr (kv_split_on) {
        pmax_l1 = experimental::CB(cb_pmax).get_write_ptr();
        xo_l1 = experimental::CB(cb_xo).get_write_ptr();
        xs_l1 = experimental::CB(cb_xs).get_write_ptr();
        split_j = get_arg_val<uint32_t>(2);
    }
    [[maybe_unused]] const auto group_core = [](uint32_t t, uint32_t l1) -> uint64_t {
        return get_noc_addr(get_arg_val<uint32_t>(3 + 2 * t), get_arg_val<uint32_t>(4 + 2 * t), l1);
    };

    for (uint32_t tok = tok_start; tok < tok_start + tok_count; ++tok) {
        uint32_t split_role = sparse_sdpa::split_role::LAST, local_chunks = 0, parts = 1;
        if constexpr (kv_split_on) {
            experimental::CB split_ctl_cb(cb_split_ctl);
            split_ctl_cb.wait_front(1);
            {
                volatile tt_l1_ptr uint32_t* sp =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(split_ctl_cb.get_read_ptr());
                split_role = sp[sparse_sdpa::split_message::ROLE];
                local_chunks = sp[sparse_sdpa::split_message::LOCAL_CHUNKS];
                parts = sp[sparse_sdpa::split_message::PARTS];
            }
            split_ctl_cb.pop_front(1);
            if (split_role == sparse_sdpa::split_role::IDLE) {
                continue;
            }
        }
        // K-gather phase: process this token's active chunks (reader signals per chunk; is_last ends the token).
        // Gather rows [0,split) on the writer's NoC into the reader-reserved destination, then ack.
        bool last = false;
        while (!last) {
            kreq_cb.wait_front(1);
            uint32_t base, split, dst_l1, scale_dst_l1 = 0;
            {
                volatile tt_l1_ptr uint32_t* request =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_read_ptr());
                base = request[sparse_sdpa::gather_request::BASE];
                split = request[sparse_sdpa::gather_request::SPLIT];
                last = request[sparse_sdpa::gather_request::IS_LAST] != 0;
                dst_l1 = request[sparse_sdpa::gather_request::DST_L1];
                if constexpr (scaled_kv) {
                    scale_dst_l1 = request[sparse_sdpa::gather_request::SCALE_DST_L1];
                }
            }
            kreq_cb.pop_front(1);
            if constexpr (scaled_kv) {
                sparse_sdpa::trid_ring_gather<
                    block_cyclic,
                    bc_chunk_local,
                    bc_sp,
                    bc_shard_stride_gap,
                    bc_slab_stride_gap,
                    sparse_sdpa::SCALED_K_TRID_RING>(
                    noc, kv, dst_l1, idx_ptr, base, 0, split, packed_row_bytes, kv_batch_page_offset);
                sparse_sdpa::scatter_packed_scales<scale_blocks>(
                    dst_l1, scale_dst_l1, 0, split, packed_row_bytes, latent_row_bytes);
            } else {
                sparse_sdpa::trid_ring_gather<
                    block_cyclic,
                    bc_chunk_local,
                    bc_sp,
                    bc_shard_stride_gap,
                    bc_slab_stride_gap,
                    sparse_sdpa::K_TRID_RING>(
                    noc, kv, dst_l1, idx_ptr, base, 0, split, k_row_bytes, kv_batch_page_offset);
            }
            kack_cb.reserve_back(1);
            kack_cb.push_back(1);
        }
        if constexpr (kv_split_on) {
            // 1. Local max over this core's chunks -> slot j of every higher participant's cb_pmax.
            if (split_j + 1 < parts) {
                experimental::CB lmax_cb(cb_lmax);
                lmax_cb.wait_front(1);
                const uint32_t src = lmax_cb.get_read_ptr();
                for (uint32_t t = split_j + 1; t < parts; ++t) {
                    noc_async_write(src, group_core(t, pmax_l1 + split_j * state_tile_bytes), state_tile_bytes);
                }
                noc_async_write_barrier();
                for (uint32_t t = split_j + 1; t < parts; ++t) {
                    noc_semaphore_inc(group_core(t, get_semaphore(sparse_sdpa::SPLIT_PMAX_SEMAPHORE)), 1);
                }
                noc_async_atomic_barrier();
                lmax_cb.pop_front(1);
            }
            // 2. Workers pass the flash state after their last chunk (slot local_chunks) to slot 0 of participant j+1.
            if (split_role == sparse_sdpa::split_role::WORKER) {
                experimental::CB done_cb(cb_split_done);
                done_cb.wait_front(1);
                noc_async_write(
                    xo_l1 + local_chunks * vDHt * state_tile_bytes,
                    group_core(split_j + 1, xo_l1),
                    vDHt * state_tile_bytes);
                noc_async_write(
                    xs_l1 + local_chunks * state_tile_bytes, group_core(split_j + 1, xs_l1), state_tile_bytes);
                noc_async_write_barrier();
                noc_semaphore_inc(group_core(split_j + 1, get_semaphore(sparse_sdpa::SPLIT_XIN_SEMAPHORE)), 1);
                noc_async_atomic_barrier();
                done_cb.pop_front(1);
                continue;  // the last participant writes this token's output
            }
        }
        out_cb.wait_front(block_tiles);  // one untilized [H, V_DIM] block
        for (uint32_t h = 0; h < H; ++h) {
            // row h (head h) at CB read-ptr byte offset h*row_bytes; DRAM page = h*S + tok.
            noc.async_write(out_cb, out, row_bytes, {.offset_bytes = h * row_bytes}, {.page_id = h * S + tok});
        }
        noc.async_write_barrier();
        out_cb.pop_front(block_tiles);
    }
}
