// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Shared-KV rows SDPA decode, writer (attn_decode/rows_sdpa.py). The stock paged SDPA decode writer
// (ttnn/.../sdpa_decode/device/kernels/dataflow/writer_decode_all.cpp) cut to one "user" per kv head, with:
//   * per-row causal masks: every chunk of this core whose last position is past the oldest row limit gets a mask
//     block (Sk tiles, bf16; tile row rho masks the positions past its verify row's limit, rows_sdpa_common.hpp),
//     pushed in chunk order; compute adds and pops one block per such chunk. The mask value is a large finite
//     negative, not -inf: a core can hold no attended position of a row (its chunks all past that row's limit), and
//     its partial must then come out finite (max = the mask value) so the tree reduction scales it by exp(-huge) = 0;
//   * the same tree reduction as the stock op;
//   * the root writes tile row r*G+i of its result to row g*G+i of verify row r's output tiles (skipped rows untouched,
//     as the stock op leaves a skipped user's output).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp"
#include "ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp"
#include "inkling_sdpa_common.hpp"

#define MAX_TREE_REDUCTION_ROUNDS 6

void kernel_main() {
    Noc noc;

    constexpr uint32_t R = get_compile_time_arg_val(0);
    constexpr uint32_t G = get_compile_time_arg_val(1);
    constexpr uint32_t vDHt = get_compile_time_arg_val(2);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t max_dynamic_chunk_size = get_compile_time_arg_val(4);
    constexpr uint32_t num_cores_per_head = get_compile_time_arg_val(5);
    constexpr uint32_t reducer_semaphore_id = get_compile_time_arg_val(6);
    constexpr uint32_t num_cores = get_compile_time_arg_val(7);
    constexpr uint32_t mask_neg_bf16 = get_compile_time_arg_val(8);
    constexpr uint32_t W = get_compile_time_arg_val(9);  // sliding window (0 = global layer)
    constexpr uint32_t num_kv_heads_w = get_compile_time_arg_val(10);
    constexpr auto out_args = TensorAccessorArgs<11>();

    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    constexpr uint32_t cb_m_in = tt::CBIndex::c_6;
    constexpr uint32_t cb_l_in = tt::CBIndex::c_7;
    constexpr uint32_t cb_cur_pos = tt::CBIndex::c_8;
    constexpr uint32_t cb_zero_in = tt::CBIndex::c_12;
    constexpr uint32_t cb_out_o = tt::CBIndex::c_16;
    constexpr uint32_t cb_out_worker = tt::CBIndex::c_16;
    constexpr uint32_t cb_out_m = tt::CBIndex::c_17;
    constexpr uint32_t cb_out_l = tt::CBIndex::c_18;
    constexpr uint32_t cb_intermed_out = tt::CBIndex::c_19;
    constexpr uint32_t cb_out = tt::CBIndex::c_20;
    constexpr uint32_t PNHt = 1;

    uint32_t arg_idx = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t group = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_num_in_reduce = get_arg_val<uint32_t>(arg_idx++);
    const bool is_tree_root = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t parent_core_in_group = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t send_at_round = get_arg_val<uint32_t>(arg_idx++);
    uint32_t children_per_round[MAX_TREE_REDUCTION_ROUNDS];
    for (uint32_t r = 0; r < MAX_TREE_REDUCTION_ROUNDS; ++r) {
        children_per_round[r] = get_arg_val<uint32_t>(arg_idx++);
    }
    tt_l1_ptr uint32_t* reduction_group_core_xs = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores_per_head;
    tt_l1_ptr uint32_t* reduction_group_core_ys = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores_per_head;
    const uint32_t n_groups = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t n_units = get_arg_val<uint32_t>(arg_idx++);

    // Semaphore encoding (stock op): each round counts in its own 4-bit field of the reducer semaphore
    constexpr uint32_t step_semaphore_inc[6] = {1, 16, 256, 4096, 65536, 1048576};
    constexpr uint32_t step_semaphore_shift[6] = {0, 4, 8, 12, 16, 20};

    // constant tiles the compute kernel keeps for the whole program: produced once
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_identity_scale_in,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
    dataflow_kernel_lib::prepare_zero_tile<cb_zero_in>();

    // Work units: unit = item * num_kv_heads + kv head. A unit is one 32-row Q tile (item = R consecutive rows of the
    // sequence x the G query heads of one kv head), computed by the num_cores_per_head cores of a group (decode: one
    // item, groups = kv heads, cores split the K span + tree reduction; prefill: one core per group, the items spread
    // over the groups round-robin, no reduction). Positions page `item` = that item's R rows; q / rq / out row r of
    // item i = sequence row i * R + r.
    for (uint32_t unit = group; unit < n_units; unit += n_groups) {
    const uint32_t cur_head = unit % num_kv_heads_w;
    const uint32_t item = unit / num_kv_heads_w;
    uint32_t raw_pos[R];
    uint32_t lim[R];
    uint32_t max_pos = 0, min_lim = 0;
    bool any = false;
    {
        CircularBuffer cb_index(cb_cur_pos);
        cb_index.wait_front(1);
        volatile tt_l1_ptr uint32_t* index_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_index.get_read_ptr());
        for (uint32_t j = 0; j < R; ++j) {
            raw_pos[j] = index_ptr[j];
        }
        cb_index.pop_front(1);
        any = rows_sdpa_limits<R>([&](uint32_t j) { return raw_pos[j]; }, lim, max_pos, min_lim);
    }
    if (!any) {
        continue;
    }

    uint32_t lo[R];
    uint32_t min_lo = max_pos, max_lo = 0;
    for (uint32_t j = 0; j < R; ++j) {
        lo[j] = inkling_lo(lim[j], W);
        min_lo = lo[j] < min_lo ? lo[j] : min_lo;
        max_lo = lo[j] > max_lo ? lo[j] : max_lo;
    }
    const uint32_t Sk_chunk_t_dynamic = get_dynamic_Sk_chunk_t<Sk_chunk_t, max_dynamic_chunk_size>(max_pos);
    const uint32_t k_chunk_size_dynamic = Sk_chunk_t_dynamic * tt::constants::TILE_HEIGHT;
    const uint32_t W_eff = W == 0 ? 0 : max_pos - min_lo + 1;
    auto [PSt, k_num_chunks, k_chunk_start, k_chunk_end, window_start_unaligned, window_start_chunk] =
        get_workload_for_core(max_pos, 0, core_num_in_reduce, num_cores_per_head, k_chunk_size_dynamic, W_eff);
    if (k_chunk_start == k_chunk_end) {
        continue;  // no data: not part of the reduction
    }

    // children with data (a child at core_num has data iff core_num < k_num_chunks)
    uint32_t num_active_children = 0;
    uint32_t active_children_per_round[MAX_TREE_REDUCTION_ROUNDS];
    uint32_t num_active_rounds = 0;
    for (uint32_t r = 0; r < MAX_TREE_REDUCTION_ROUNDS; ++r) {
        const uint32_t child_id = children_per_round[r];
        if (child_id != UINT32_MAX && child_id < k_num_chunks) {
            active_children_per_round[r] = child_id;
            num_active_children++;
            num_active_rounds = r + 1;
        } else {
            active_children_per_round[r] = UINT32_MAX;
        }
    }
    const bool should_send_to_parent = (parent_core_in_group != UINT32_MAX);


    // per-row causal masks, one Sk-tile block per masked chunk, in chunk order
    constexpr uint32_t mask_tile_bytes = get_tile_size(cb_mask_in);
    static_assert(mask_tile_bytes == 2048, "bf16 32x32 mask tiles");
    constexpr uint32_t NEG2 = mask_neg_bf16 | (mask_neg_bf16 << 16);
    CircularBuffer cb_mask(cb_mask_in);
    for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
        if (!inkling_chunk_masked(k_chunk, k_chunk_size_dynamic, min_lim, W, max_lo)) {
            continue;
        }
        cb_mask.reserve_back(Sk_chunk_t_dynamic);
        volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_mask.get_write_ptr());
        for (uint32_t t = 0; t < Sk_chunk_t_dynamic; ++t) {
            const uint32_t base = (k_chunk * Sk_chunk_t_dynamic + t) * 32;  // first position of this tile
            if (base + 31 <= min_lim && base >= max_lo) {  // every row attends the whole tile
                fill_tile_zeros<mask_tile_bytes, false>(noc, cb_mask_in, t);
            } else if (base > max_pos || base + 31 < min_lo) {  // no row attends any of it
                for (uint32_t w = 0; w < mask_tile_bytes / 4; ++w) {
                    m[w] = NEG2;
                }
            } else {
                for (uint32_t rho = 0; rho < 32; ++rho) {
                    const uint32_t l = rho < R * G ? lim[rho / G] : max_pos;
                    const uint32_t lw = rho < R * G ? lo[rho / G] : min_lo;
                    // row rho attends columns [c_lo, n) of this tile (positions [lo, lim]); the rest get the mask value
                    const uint32_t n = l < base ? 0 : (l - base >= 31 ? 32 : l - base + 1);
                    const uint32_t c_lo = lw <= base ? 0 : (lw - base > 32 ? 32 : lw - base);
                    volatile tt_l1_ptr uint32_t* row = m + (rho >> 4) * 256 + (rho & 15) * 8;  // uint32 words
                    for (uint32_t half = 0; half < 2; ++half) {
                        for (uint32_t w = 0; w < 8; ++w) {
                            const uint32_t c0 = half * 16 + 2 * w;
                            const uint32_t lo16 = (c0 < n && c0 >= c_lo) ? 0u : mask_neg_bf16;
                            const uint32_t hi16 = (c0 + 1 < n && c0 + 1 >= c_lo) ? 0u : mask_neg_bf16;
                            row[half * 128 + w] = lo16 | (hi16 << 16);
                        }
                    }
                }
            }
            m += mask_tile_bytes / 4;
        }
        noc.write_zeros_l1_barrier();
        cb_mask.push_back(Sk_chunk_t_dynamic);
    }

    constexpr uint32_t out_chunk_tiles = PNHt * vDHt;
    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    volatile tt_l1_ptr uint32_t* in0_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(reducer_semaphore_id));
    noc.async_write_barrier();  // #19201 BH hang workaround (stock op)

    // tree reduction: receive each round's child (l, m, o), for compute to combine
    if (num_active_children > 0) {
        for (uint32_t round = 0; round < num_active_rounds; ++round) {
            if (active_children_per_round[round] == UINT32_MAX) {
                continue;
            }
            while (true) {
                invalidate_l1_cache();
                const uint32_t sem_val = *in0_receiver_semaphore_addr_ptr;
                if (((sem_val >> step_semaphore_shift[round]) & 0x0F) >= 1) {
                    break;
                }
            }
            constexpr uint32_t tile_bytes_intermed = get_tile_size(cb_intermed_out);
            constexpr uint32_t o_read_size = out_chunk_tiles * tile_bytes_intermed;
            constexpr uint32_t ml_read_size = PNHt * tile_bytes_intermed;
            const uint32_t block_offset = round * (out_chunk_tiles + 2 * PNHt) * tile_bytes_intermed;
            CircularBuffer cb_intermed(cb_intermed_out);
            const uint8_t noc_id = noc.get_noc_id();
            const uint32_t my_noc_x = my_x[noc_id];
            const uint32_t my_noc_y = my_y[noc_id];
            uint32_t intermed_l1_read_addr = cb_intermed.get_read_ptr() + block_offset;
            UnicastEndpoint intermed_src;

            CircularBuffer cb_l(cb_l_in);
            cb_l.reserve_back(PNHt);
            noc.async_read(
                intermed_src,
                CoreLocalMem<uint32_t>(cb_l.get_write_ptr()),
                ml_read_size,
                {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = intermed_l1_read_addr},
                {});
            intermed_l1_read_addr += ml_read_size;
            noc.async_read_barrier();
            cb_l.push_back(PNHt);

            CircularBuffer cb_m(cb_m_in);
            cb_m.reserve_back(PNHt);
            noc.async_read(
                intermed_src,
                CoreLocalMem<uint32_t>(cb_m.get_write_ptr()),
                ml_read_size,
                {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = intermed_l1_read_addr},
                {});
            intermed_l1_read_addr += ml_read_size;
            noc.async_read_barrier();
            cb_m.push_back(PNHt);

            CircularBuffer cb_o(cb_out_o);
            cb_o.reserve_back(out_chunk_tiles);
            noc.async_read(
                intermed_src,
                CoreLocalMem<uint32_t>(cb_o.get_write_ptr()),
                o_read_size,
                {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = intermed_l1_read_addr},
                {});
            noc.async_read_barrier();
            cb_o.push_back(out_chunk_tiles);
        }
    }

    if (!is_tree_root) {
        if (!should_send_to_parent) {
            continue;
        }
        // send (l, m, o) to the parent's round slot, then bump its round counter
        CircularBuffer cb_out_w(cb_out_worker);
        CircularBuffer cb_out_m_buf(cb_out_m);
        CircularBuffer cb_out_l_buf(cb_out_l);
        CircularBuffer cb_intermed(cb_intermed_out);
        cb_out_w.wait_front(out_chunk_tiles);
        cb_out_m_buf.wait_front(PNHt);
        cb_out_l_buf.wait_front(PNHt);
        constexpr uint32_t w_tile_bytes = get_tile_size(cb_out_worker);
        const uint32_t block_offset = send_at_round * (out_chunk_tiles + 2 * PNHt) * w_tile_bytes;
        constexpr uint32_t o_write_size = out_chunk_tiles * w_tile_bytes;
        constexpr uint32_t ml_write_size = PNHt * w_tile_bytes;
        const uint32_t parent_noc_x = reduction_group_core_xs[parent_core_in_group];
        const uint32_t parent_noc_y = reduction_group_core_ys[parent_core_in_group];
        uint32_t output_write_addr = cb_intermed.get_write_ptr() + block_offset;
        UnicastEndpoint output_dst;
        noc.async_write(
            CoreLocalMem<uint32_t>(cb_out_l_buf.get_read_ptr()),
            output_dst,
            ml_write_size,
            {},
            {.noc_x = parent_noc_x, .noc_y = parent_noc_y, .addr = output_write_addr});
        output_write_addr += ml_write_size;
        noc.async_write(
            CoreLocalMem<uint32_t>(cb_out_m_buf.get_read_ptr()),
            output_dst,
            ml_write_size,
            {},
            {.noc_x = parent_noc_x, .noc_y = parent_noc_y, .addr = output_write_addr});
        output_write_addr += ml_write_size;
        noc.async_write(
            CoreLocalMem<uint32_t>(cb_out_w.get_read_ptr()),
            output_dst,
            o_write_size,
            {},
            {.noc_x = parent_noc_x, .noc_y = parent_noc_y, .addr = output_write_addr});
        noc.async_write_barrier();
        Semaphore<>(reducer_semaphore_id).up(noc, parent_noc_x, parent_noc_y, step_semaphore_inc[send_at_round]);
        cb_out_w.pop_front(out_chunk_tiles);
        cb_out_m_buf.pop_front(PNHt);
        cb_out_l_buf.pop_front(PNHt);
        noc.async_atomic_barrier();
        continue;
    }

    // root: tile row r*G+i of each result tile -> row cur_head*G+i of verify row r's output tile (two face rows)
    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<tile_bytes, num_cores>();
    uint32_t barrier_count = 0;
    static_assert(tile_bytes == 2048, "bf16 output tiles");
    const auto out_writer = TensorAccessor(out_args, out_addr, tile_bytes);
    CircularBuffer cb_out_buf(cb_out);
    cb_out_buf.wait_front(out_chunk_tiles);
    const uint32_t res_l1 = cb_out_buf.get_read_ptr();
    for (uint32_t d = 0; d < vDHt; ++d) {
        for (uint32_t r = 0; r < R; ++r) {
            if (static_cast<int32_t>(raw_pos[r]) < 0) {
                continue;
            }
            const uint32_t src_tile = res_l1 + d * tile_bytes;
            rows_sdpa_row_runs(r * G, cur_head * G, G, [&](uint32_t s, uint32_t t, uint32_t k) {
                for (uint32_t half = 0; half < 2; ++half) {
                    noc.async_write(
                        CoreLocalMem<uint32_t>(src_tile + rows_sdpa_row_off(s) + half * 512),
                        out_writer,
                        k * 32,
                        {},
                        {.page_id = (item * R + r) * vDHt + d, .offset_bytes = rows_sdpa_row_off(t) + half * 512});
                    if (++barrier_count == barrier_threshold) {
                        noc.async_writes_flushed();
                        barrier_count = 0;
                    }
                }
            });
        }
    }
    noc.async_write_barrier();
    cb_out_buf.pop_front(out_chunk_tiles);
    }  // units
}
