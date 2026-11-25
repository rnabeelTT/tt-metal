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
#ifndef SDPA_PAIR
#define SDPA_PAIR 0
#endif
#ifndef SDPA_MI
#define SDPA_MI 1  // prefill: consecutive items per unit (no tree reduction then)
#endif
#ifndef SDPA_POS_WORDS
#define SDPA_POS_WORDS 16  // the cur_pos stick in uint32 words
#endif

#ifndef SDPA_FUSE
#define SDPA_FUSE 0
#endif

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
    // the bias-table tiles (read here, beside the masks: the reader's NOC streams K / V meanwhile)
    constexpr uint32_t E = get_compile_time_arg_val(out_args.next_compile_time_args_offset());
    constexpr uint32_t M_tiles = get_compile_time_arg_val(out_args.next_compile_time_args_offset() + 1);
    constexpr auto tbl_args = TensorAccessorArgs<out_args.next_compile_time_args_offset() + 2>();
    constexpr uint32_t cb_tbl = tt::CBIndex::c_11;  // per band chunk and item: R x Sk table tiles (row-major [j][t])

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
    constexpr uint32_t MI = SDPA_MI;
    // SDPA_PAIR (prefill): rows 2p / 2p + 1 (consecutive positions) read one pair-table tile P[phi][mm] (M + 1 per
    // phase): rows 0-15 = T[phi][mm - 1], rows 16-31 = row 2p + 1's T[phi + 1][mm - 1] (phi 31: T[0][mm]), mm = a - b + 1
    constexpr uint32_t RP = SDPA_PAIR ? R / 2 : R;
    constexpr uint32_t RS = SDPA_PAIR ? 2 : 1;
    constexpr uint32_t MO = SDPA_PAIR ? 1 : 0;

    uint32_t arg_idx = 0;
#ifdef SDPA_CARGS  // (prefill: the addresses in the common runtime args)
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
#else
    const uint32_t out_addr = get_arg_val<uint32_t>(arg_idx++);
#endif
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
#ifdef SDPA_CARGS
    const uint32_t tbl_addr = get_common_arg_val<uint32_t>(1);
#else
    const uint32_t tbl_addr = get_arg_val<uint32_t>(arg_idx++);
#endif

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
#ifdef SDPA_NEG_OFF
    // a fully masked bf16 tile, built while the positions are not yet known (past the broadcast slots of the scratch
    // CB): the masks copy it (one L1 -> L1 read) instead of 512 RISC stores on the bias phase's path
    uint32_t neg_l1 = 0;
    {
        CircularBuffer cb_scr(tt::CBIndex::c_10);
        neg_l1 = ((cb_scr.get_write_ptr() + 63) & ~63u) + SDPA_NEG_OFF;
        constexpr uint32_t NEGW = mask_neg_bf16 | (mask_neg_bf16 << 16);
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(neg_l1);
        for (uint32_t w = 0; w < 512; ++w) {
            p[w] = NEGW;
        }
    }
#endif

    // Work units: unit = item * num_kv_heads + kv head. A unit is one 32-row Q tile (item = R consecutive rows of the
    // sequence x the G query heads of one kv head), computed by the num_cores_per_head cores of a group (decode: one
    // item, groups = kv heads, cores split the K span + tree reduction; prefill: one core per group, the items spread
    // over the groups round-robin, no reduction). Positions page `item` = that item's R rows; q / rq / out row r of
    // item i = sequence row i * R + r.
    for (uint32_t unit = group; unit < n_units; unit += n_groups) {
    const uint32_t cur_head = unit % num_kv_heads_w;
    const uint32_t item = (unit / num_kv_heads_w) * MI;  // the unit's first item
    uint32_t raw_pos[MI * R];
    uint32_t lim[MI * R];
    uint32_t max_pos = 0, min_lim = 0;
    bool any = false;
    {
        CircularBuffer cb_index(cb_cur_pos);
        cb_index.wait_front(MI);
        volatile tt_l1_ptr uint32_t* index_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_index.get_read_ptr());
        for (uint32_t j = 0; j < MI * R; ++j) {
            raw_pos[j] = index_ptr[(j / R) * SDPA_POS_WORDS + j % R];
        }
        cb_index.pop_front(MI);
        any = rows_sdpa_limits<MI * R>([&](uint32_t j) { return raw_pos[j]; }, lim, max_pos, min_lim);
    }
    if (!any) {
        continue;
    }

    uint32_t lo[MI * R];
    uint32_t min_lo = max_pos, max_lo = 0;
    for (uint32_t j = 0; j < MI * R; ++j) {
        lo[j] = inkling_lo(lim[j], W);
        min_lo = lo[j] < min_lo ? lo[j] : min_lo;
        max_lo = lo[j] > max_lo ? lo[j] : max_lo;
    }
    const uint32_t Sk_chunk_t_dynamic = get_dynamic_Sk_chunk_t<Sk_chunk_t, max_dynamic_chunk_size>(max_pos);
    const uint32_t k_chunk_size_dynamic = Sk_chunk_t_dynamic * tt::constants::TILE_HEIGHT;
    const uint32_t W_eff = W == 0 ? 0 : max_pos - min_lo + 1;
    uint32_t cores_eff;
    auto [PSt, k_num_chunks, k_chunk_start, k_chunk_end, window_start_unaligned, window_start_chunk] =
        inkling_workload(max_pos, core_num_in_reduce, num_cores_per_head, k_chunk_size_dynamic, W_eff, cores_eff);
    if (k_chunk_start == k_chunk_end) {
        continue;  // no data: not part of the reduction
    }

    // children with data (a child at core_num has data iff core_num < k_num_chunks)
    uint32_t num_active_children = 0;
    uint32_t active_children_per_round[MAX_TREE_REDUCTION_ROUNDS];
    uint32_t num_active_rounds = 0;
    for (uint32_t r = 0; r < MAX_TREE_REDUCTION_ROUNDS; ++r) {
        const uint32_t child_id = children_per_round[r];
        if (child_id != UINT32_MAX && child_id < k_num_chunks && child_id < cores_eff) {
            active_children_per_round[r] = child_id;
            num_active_children++;
            num_active_rounds = r + 1;
        } else {
            active_children_per_round[r] = UINT32_MAX;
        }
    }
    const bool should_send_to_parent = (parent_core_in_group != UINT32_MAX);


    // per-row causal masks, one Sk-tile block per masked chunk and item, in chunk order
    constexpr uint32_t mask_tile_bytes = get_tile_size(cb_mask_in);
    static_assert(mask_tile_bytes == 2048, "bf16 32x32 mask tiles");
    constexpr uint32_t NEG2 = mask_neg_bf16 | (mask_neg_bf16 << 16);
    CircularBuffer cb_mask(cb_mask_in);
    CircularBuffer cb_tblc(cb_tbl);
    const auto tbl_reader = TensorAccessor(tbl_args, tbl_addr, mask_tile_bytes);
    for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
        uint32_t tb, Skc;
        inkling_chunk_tiles(
            k_chunk, Sk_chunk_t_dynamic, max_pos, min_lo, k_chunk + 1 == k_chunk_end, tb, Skc, k_num_chunks,
            window_start_chunk, cores_eff);
        const bool masked = inkling_tiles_masked(tb, Skc, min_lim, W, max_lo);
        const bool band = inkling_tiles_band(tb, Skc, min_lim, E);
        for (uint32_t mi = 0; mi < MI && (masked || band); ++mi) {
        if (band) {
            // R x Sk table tiles: row j, key tile b = tb + t -> T[phi_j][m = a_j - b] (zero outside [0, M))
            const uint32_t n = RP * Skc;
            cb_tblc.reserve_back(n);
            uint32_t idx = 0;
            for (uint32_t j = 0; j < R; j += RS) {
                const uint32_t a = (lim[mi * R + j] >> 5) + MO, phi = lim[mi * R + j] & 31;
                for (uint32_t t = 0; t < Skc; ++t, ++idx) {
                    const uint32_t b = tb + t;
                    if (b <= a && a - b < M_tiles + MO) {
                        noc_async_read(
                            tbl_reader.get_noc_addr(phi * (M_tiles + MO) + (a - b)),
                            cb_tblc.get_write_ptr() + idx * mask_tile_bytes,
                            mask_tile_bytes);
                    } else {
                        fill_tile_zeros<mask_tile_bytes, false>(noc, cb_tbl, idx);
                    }
                }
            }
            if (!masked) {
                noc.write_zeros_l1_barrier();
                noc_async_read_barrier();
                cb_tblc.push_back(n);
            }
        }
        if (!masked) {
            continue;
        }  // (a masked chunk is a band chunk: its table reads land while the mask is built, both pushed below)
        cb_mask.reserve_back(Skc);
        volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_mask.get_write_ptr());
        // a fully masked tile is written once per block (RISC stores) and copied (L1 -> L1 NOC reads) to the others;
        // a partly masked tile: zeros (NOC), then the live rows (R x G per item: the rows past them are never read)
#ifdef SDPA_NEG_OFF
        uint32_t neg_tile = neg_l1;
#else
        uint32_t neg_tile = 0;
#endif
        for (uint32_t t = 0; t < Skc; ++t) {
            const uint32_t base = (tb + t) * 32;  // first position of this tile
            if (base + 31 <= min_lim && base >= max_lo) {  // every row attends the whole tile
                fill_tile_zeros<mask_tile_bytes, false>(noc, cb_mask_in, t);
            } else if (base > max_pos || base + 31 < min_lo) {  // no row attends any of it
                const uint32_t dst = reinterpret_cast<uint32_t>(m);
                if (neg_tile == 0) {
                    for (uint32_t w = 0; w < mask_tile_bytes / 4; ++w) {
                        m[w] = NEG2;
                    }
                    neg_tile = dst;
                } else {
                    noc_async_read(get_noc_addr(neg_tile), dst, mask_tile_bytes);
                }
            } else {
                fill_tile_zeros<mask_tile_bytes, false>(noc, cb_mask_in, t);
                noc.write_zeros_l1_barrier();
                for (uint32_t rho = 0; rho < (R * G < 32 ? R * G : 32); ++rho) {
                    const uint32_t l = rho < R * G ? lim[mi * R + rho / G] : max_pos;
                    const uint32_t lw = rho < R * G ? lo[mi * R + rho / G] : min_lo;
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
        noc_async_read_barrier();
        if (band) {
            cb_tblc.push_back(RP * Skc);
        }
        cb_mask.push_back(Skc);
        }  // items
    }

    constexpr uint32_t out_chunk_tiles = PNHt * vDHt;
    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    volatile tt_l1_ptr uint32_t* in0_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(reducer_semaphore_id));
    noc.async_write_barrier();  // #19201 BH hang workaround (stock op)

    // tree reduction: receive each round's child (l, m, o), for compute to combine (decode only, MI == 1)
    if (MI == 1 && num_active_children > 0) {
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
            CircularBuffer cb_m(cb_m_in);
            CircularBuffer cb_o(cb_out_o);
            cb_l.reserve_back(PNHt);
            noc.async_read(
                intermed_src,
                CoreLocalMem<uint32_t>(cb_l.get_write_ptr()),
                ml_read_size,
                {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = intermed_l1_read_addr},
                {});
            intermed_l1_read_addr += ml_read_size;
#if !SDPA_FUSE
            noc.async_read_barrier();
            cb_l.push_back(PNHt);
#endif
            cb_m.reserve_back(PNHt);
            noc.async_read(
                intermed_src,
                CoreLocalMem<uint32_t>(cb_m.get_write_ptr()),
                ml_read_size,
                {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = intermed_l1_read_addr},
                {});
            intermed_l1_read_addr += ml_read_size;
#if !SDPA_FUSE
            noc.async_read_barrier();
            cb_m.push_back(PNHt);
#endif
            cb_o.reserve_back(out_chunk_tiles);
            noc.async_read(
                intermed_src,
                CoreLocalMem<uint32_t>(cb_o.get_write_ptr()),
                o_read_size,
                {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = intermed_l1_read_addr},
                {});
            noc.async_read_barrier();  // (SDPA_FUSE: one barrier for the child's l, m and o)
#if SDPA_FUSE
            cb_l.push_back(PNHt);
            cb_m.push_back(PNHt);
#endif
            cb_o.push_back(out_chunk_tiles);
        }
    }

    if constexpr (MI == 1) {  // (MI > 1: every unit's core is a root)
    if (!is_tree_root) {
        if (!should_send_to_parent) {
            continue;
        }
        // send (l, m, o) to the parent's round slot, then bump its round counter
#if SDPA_FUSE
        // the compute's final OUT_ACC and running max / sum CBs (no copies; they swap roles once per chunk and per
        // merged child: an odd count leaves them in max_1 / sum_1)
        const bool odd = ((k_chunk_end - k_chunk_start) + num_active_children) & 1;
        CircularBuffer cb_done(cb_out_l);  // (the compute's token: its last pack is done)
        cb_done.wait_front(1);
        CircularBuffer cb_out_w(tt::CBIndex::c_26);
        CircularBuffer cb_out_m_buf(odd ? tt::CBIndex::c_27 : tt::CBIndex::c_28);
        CircularBuffer cb_out_l_buf(odd ? tt::CBIndex::c_29 : tt::CBIndex::c_30);
#else
        CircularBuffer cb_out_w(cb_out_worker);
        CircularBuffer cb_out_m_buf(cb_out_m);
        CircularBuffer cb_out_l_buf(cb_out_l);
#endif
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
#if SDPA_FUSE
        cb_done.pop_front(1);
#endif
        noc.async_atomic_barrier();
        continue;
    }
    }

    // root: tile row r*G+i of each result tile -> row cur_head*G+i of verify row r's output tile (two face rows)
    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<tile_bytes, num_cores>();
    uint32_t barrier_count = 0;
    static_assert(tile_bytes == 2048, "bf16 output tiles");
    const auto out_writer = TensorAccessor(out_args, out_addr, tile_bytes);
    CircularBuffer cb_out_buf(cb_out);
    cb_out_buf.wait_front(MI * out_chunk_tiles);
    const uint32_t res_l1 = cb_out_buf.get_read_ptr();
    if constexpr (MI * R == 32) {
        // the unit's 32 rows are one tile row of the [1, 1, T, NH * HD] output: whole tiles for its G heads, assembled
        // in L1 (tile row r*G+i of result tile (mi, d) -> row mi*R + r of head i's tile d), then written as tiles
        const uint8_t noc_id = noc.get_noc_id();
        const uint32_t my_noc_x = my_x[noc_id], my_noc_y = my_y[noc_id];
        UnicastEndpoint self;
        CircularBuffer cb_stage(cb_intermed_out);
        cb_stage.reserve_back(G * vDHt);
        const uint32_t st_l1 = cb_stage.get_write_ptr();
        for (uint32_t i = 0; i < G; ++i) {
            for (uint32_t d = 0; d < vDHt; ++d) {
                const uint32_t dst_tile = st_l1 + (i * vDHt + d) * tile_bytes;
                for (uint32_t mi = 0; mi < MI; ++mi) {
                    const uint32_t src_tile = res_l1 + (mi * vDHt + d) * tile_bytes;
                    for (uint32_t r = 0; r < R; ++r) {
                        for (uint32_t half = 0; half < 2; ++half) {
                            noc.async_read(
                                self,
                                CoreLocalMem<uint32_t>(dst_tile + rows_sdpa_row_off(mi * R + r) + half * 512),
                                32,
                                {.noc_x = my_noc_x,
                                 .noc_y = my_noc_y,
                                 .addr = src_tile + rows_sdpa_row_off(r * G + i) + half * 512},
                                {});
                        }
                    }
                }
            }
        }
        noc.async_read_barrier();
        const uint32_t row_tile = item / MI, tiles_per_row = num_kv_heads_w * G * vDHt;
        for (uint32_t i = 0; i < G; ++i) {
            for (uint32_t d = 0; d < vDHt; ++d) {
                noc.async_write(
                    CoreLocalMem<uint32_t>(st_l1 + (i * vDHt + d) * tile_bytes),
                    out_writer,
                    tile_bytes,
                    {},
                    {.page_id = row_tile * tiles_per_row + (cur_head * G + i) * vDHt + d});
            }
        }
        noc.async_write_barrier();
        cb_out_buf.pop_front(MI * out_chunk_tiles);
        continue;
    }
    for (uint32_t mi = 0; mi < MI; ++mi) {
    for (uint32_t d = 0; d < vDHt; ++d) {
        for (uint32_t r = 0; r < R; ++r) {
            if (static_cast<int32_t>(raw_pos[mi * R + r]) < 0) {
                continue;
            }
            const uint32_t src_tile = res_l1 + (mi * vDHt + d) * tile_bytes;
            rows_sdpa_row_runs(r * G, cur_head * G, G, [&](uint32_t s, uint32_t t, uint32_t k) {
                for (uint32_t half = 0; half < 2; ++half) {
                    noc.async_write(
                        CoreLocalMem<uint32_t>(src_tile + rows_sdpa_row_off(s) + half * 512),
                        out_writer,
                        k * 32,
                        {},
                        {.page_id = ((item + mi) * R + r) * vDHt + d,
                         .offset_bytes = rows_sdpa_row_off(t) + half * 512});
                    if (++barrier_count == barrier_threshold) {
                        noc.async_writes_flushed();
                        barrier_count = 0;
                    }
                }
            });
        }
    }
    }  // items
    noc.async_write_barrier();
    cb_out_buf.pop_front(MI * out_chunk_tiles);
    }  // units
}
