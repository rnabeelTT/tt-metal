// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Inkling SDPA decode, compute (tt/attn/inkling_sdpa.py): the rows SDPA decode compute (qwen36 attn_decode, below) plus
// Inkling's relative-position bias: each band chunk first computes BIAS = sum_j Rq_j @ T[phi_j][m_j] (one small matmul
// over the R verify rows, the writer's causal / window mask folded in on DST) and the QK matmul adds that block.
//
// Shared-KV rows SDPA decode, compute (attn_decode/rows_sdpa.py). The stock paged SDPA decode compute
// (ttnn/.../sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp: flash-decode over this core's K chunks, then the
// tree reduction) for one 32-row Q tile (the verify rows' query heads of one kv head), with the stock op's data formats
// and helpers. One change: the stock op adds its causal mask on the last chunk only (one position per user); here
// every chunk whose last position is past the oldest row limit adds (and pops) the writer's per-row mask block for it
// (rows_sdpa_common.hpp), so rows at different positions, straddling a chunk or a core boundary, each see their own
// causal limit.

#define LLK_ZEROFLAG_OUTLINE 1

#include <cstdint>

#include "tools/profiler/kernel_profiler.hpp"

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)
#define MAX_TREE_REDUCTION_ROUNDS 6

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"
#include "inkling_sdpa_common.hpp"

#ifndef SDPA_MI
#define SDPA_MI 1
#endif
#ifndef SDPA_FIN2
#define SDPA_FIN2 0
#endif
#ifndef SDPA_FUSE
#define SDPA_FUSE 0
#endif

ALWI void swap_cbs(uint32_t& a, uint32_t& b) {
    const uint32_t t = a;
    a = b;
    b = t;
}

// SDPA_FUSE helpers (one Q tile row; bf16 DST and CBs, so values kept in DST equal their packed + unpacked copies)
// out0 = exp((a0 - b) * scale), out1 = exp((a1 - b) * scale): the two sub_exp_block steps in one DST round
template <uint32_t scale_fp32>
void sub_exp_pair(uint32_t a0, uint32_t a1, uint32_t b, uint32_t out0, uint32_t out1) {
    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;
    sub_init(a0, b);
    exp_tile_init<EXP_APPROX_MODE>();
    CircularBuffer(a0).wait_front(1);
    CircularBuffer(a1).wait_front(1);
    CircularBuffer(b).wait_front(1);
    CircularBuffer(out0).reserve_back(1);
    CircularBuffer(out1).reserve_back(1);
    tile_regs_acquire();
    sub_tiles(a0, b, 0, 0, 0);
    sub_tiles(a1, b, 0, 0, 1);
    MATH((exp_tile_first_column<EXP_APPROX_MODE, scale_bf16>(0)));
    MATH((exp_tile_first_column<EXP_APPROX_MODE, scale_bf16>(1)));
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out0);
    pack_tile(1, out1);
    tile_regs_release();
    CircularBuffer(out0).push_back(1);
    CircularBuffer(out1).push_back(1);
}

// sum = 1 / (sum * emd + emd2) (the product as srcA of the add, as add_block_inplace(sum, emd2)); o *= emd (bcast
// cols); emd / emd2 popped
template <uint32_t cols>
void fin_sum_recip_scale(uint32_t sum, uint32_t emd, uint32_t emd2, uint32_t o) {
    CircularBuffer cb_sum(sum), cb_o(o);
    reconfig_data_format(sum, emd);
    pack_reconfig_data_format(sum);
    cb_sum.wait_front(1);
    CircularBuffer(emd).wait_front(1);
    CircularBuffer(emd2).wait_front(1);
    cb_o.wait_front(cols);
    tile_regs_acquire();
    mul_init(sum, emd);
    mul_tiles(sum, emd, 0, 0, 0);
    add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(emd2);
    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(emd2, 0, 0);
    recip_tile_init();
    MATH((recip_tile_first_column(0)));
    mul_bcast_cols_init(o, emd);
    for (uint32_t j = 0; j < cols; ++j) {
        mul_tiles_bcast_cols(o, emd, j, 0, 1 + j);
    }
    tile_regs_commit();
    cb_sum.pop_front(1);
    cb_o.pop_front(cols);
    cb_sum.reserve_back(1);
    cb_o.reserve_back(cols);
    tile_regs_wait();
    pack_tile(0, sum);
    for (uint32_t j = 0; j < cols; ++j) {
        pack_tile(1 + j, o);
    }
    tile_regs_release();
    cb_sum.push_back(1);
    cb_o.push_back(cols);
    CircularBuffer(emd).pop_front(1);
    CircularBuffer(emd2).pop_front(1);
}

// acc += child * emd2 (bcast cols; the product as srcB of the add, as add_block_inplace(acc, im2)); child / emd2 popped
template <uint32_t cols>
void acc_add_scaled(uint32_t acc, uint32_t child, uint32_t emd2) {
    CircularBuffer cb_acc(acc), cb_child(child);
    reconfig_data_format(child, emd2);
    pack_reconfig_data_format(acc);
    cb_acc.wait_front(cols);
    cb_child.wait_front(cols);
    CircularBuffer(emd2).wait_front(1);
    tile_regs_acquire();
    mul_bcast_cols_init(child, emd2);
    for (uint32_t j = 0; j < cols; ++j) {
        mul_tiles_bcast_cols(child, emd2, j, 0, j);
    }
    add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(acc);
    for (uint32_t j = 0; j < cols; ++j) {
        add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(acc, j, j);
    }
    tile_regs_commit();
    cb_acc.pop_front(cols);
    cb_child.pop_front(cols);
    cb_acc.reserve_back(cols);
    tile_regs_wait();
    for (uint32_t j = 0; j < cols; ++j) {
        pack_tile(j, acc);
    }
    tile_regs_release();
    cb_acc.push_back(cols);
    CircularBuffer(emd2).pop_front(1);
}

// matmul_blocks (stock compute_common.hpp) for one 1 x N subblock per in0 tile row (subblock_h 1, one in1 subblock,
// one K block), with in0 starting at tile in0_off and the added mask block offset by row * mask_row_stride: the band
// bias of item m (in0 = its R Rq tiles) and the QK of MI stacked Q tile rows (each row adds its own bias row).
ALWI void inkling_matmul_rows(
    uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t M, uint32_t N, uint32_t K, bool transpose,
    bool add_mask, uint32_t mask_cb, uint32_t zero_cb, uint32_t in0_off, uint32_t mask_row_stride) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_out(out_cb);
    CircularBuffer cb_mask(mask_cb);
    CircularBuffer cb_zero(zero_cb);
    matmul_block_init(in0_cb, in1_cb, transpose, N, 1, K);
    reconfig_data_format(in1_cb, in0_cb);
    cb_in1.wait_front(K * N);
    for (uint32_t row = 0; row < M; ++row) {
        cb_out.reserve_back(N);
        cb_in0.wait_front(in0_off + (row + 1) * K);
        tile_regs_acquire();
        uint32_t in0_index = in0_off + row * K, in1_index = 0;
        for (uint32_t k = 0; k < K; ++k) {
            matmul_block(in0_cb, in1_cb, in0_index, in1_index, 0, transpose, N, 1, K);
            in0_index++;
            in1_index += N;
        }
        if (add_mask) {
            const uint32_t m0 = row * mask_row_stride;
            cb_mask.wait_front(m0 + N);
            cb_zero.wait_front(1);
            reconfig_data_format(zero_cb, mask_cb);
            add_init(zero_cb, mask_cb, true);
            for (uint32_t i = 0; i < N; i++) {
                add_tiles(zero_cb, mask_cb, 0, m0 + i, i);
            }
            reconfig_data_format(in1_cb, in0_cb);
            matmul_block_init(in0_cb, in1_cb, transpose, N, 1, K);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t c = 0; c < N; c++) {
            pack_tile<true>(c, out_cb, c);  // (relative to the write pointer: each row is pushed as it completes)
        }
        tile_regs_release();
        cb_out.push_back(N);
    }
    cb_in1.pop_front(K * N);
}

void kernel_main() {
    constexpr uint32_t DHt = get_compile_time_arg_val(0);
    constexpr uint32_t vDHt = get_compile_time_arg_val(1);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(2);
    constexpr uint32_t max_dynamic_chunk_size = get_compile_time_arg_val(3);
    constexpr uint32_t num_cores_per_head = get_compile_time_arg_val(4);
    constexpr uint32_t R = get_compile_time_arg_val(5);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(6);
    constexpr uint32_t dst_size = get_compile_time_arg_val(7);
    constexpr uint32_t W = get_compile_time_arg_val(8);  // sliding window (0 = global layer)
    constexpr uint32_t E = get_compile_time_arg_val(9);  // rel_extent: bias band
    constexpr uint32_t G = get_compile_time_arg_val(10);  // q heads per kv head (a unit's tile rows: MI x R x G)

    // SDPA_MI (prefill): MI consecutive items per unit, their Q tiles stacked as MI tile rows: each K / V chunk is read
    // once for all of them (per-row max / sum as in the stock prefill SDPA; band bias and masks per item)
    constexpr uint32_t MI = SDPA_MI;
    constexpr uint32_t Sq_chunk_t = MI;
    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;
    // P . V: one tile row, out subblock = min(vDHt, dst) wide (the stock factory's out_* configuration)
    constexpr uint32_t out_subblock_w = vDHt < dst_size ? vDHt : dst_size;
    constexpr uint32_t out_subblock_h = 1;
    constexpr uint32_t out_in0_num_subblocks = Sq_chunk_t;
    constexpr uint32_t out_in1_num_subblocks = vDHt / out_subblock_w;
    static_assert(vDHt % out_subblock_w == 0, "V head dim tiles split into whole subblocks");
    // the softmax SFPU steps on the live Q rows only: decode / verify units hold R x G <= 16 rows (faces 0 / 1:
    // VectorMode::R); rows are independent through QK, the row softmax and PV, so the live rows are unchanged
    constexpr VectorMode vector_mode = (MI * R * G <= 16) ? VectorMode::R : VectorMode::RC;

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    constexpr uint32_t cb_m_in = tt::CBIndex::c_6;
    constexpr uint32_t cb_l_in = tt::CBIndex::c_7;
    constexpr uint32_t cb_zero_in = tt::CBIndex::c_12;
    constexpr uint32_t cb_cur_pos = tt::CBIndex::c_15;
    constexpr uint32_t cb_qk_im = tt::CBIndex::c_24;
    constexpr uint32_t cb_out_im = tt::CBIndex::c_25;
    constexpr uint32_t cb_out_accumulate_im = tt::CBIndex::c_26;
    constexpr uint32_t cb_max_1 = tt::CBIndex::c_27;
    constexpr uint32_t cb_max_2 = tt::CBIndex::c_28;
    constexpr uint32_t cb_sum_1 = tt::CBIndex::c_29;
    constexpr uint32_t cb_sum_2 = tt::CBIndex::c_30;
    constexpr uint32_t cb_exp_max_diff = tt::CBIndex::c_31;
    constexpr uint32_t cb_prev_sum_2 = tt::CBIndex::c_21;
    constexpr uint32_t cb_exp_max_diff_2 = tt::CBIndex::c_22;
    constexpr uint32_t cb_rq = tt::CBIndex::c_4;    // R tiles (reader)
    constexpr uint32_t cb_tbl = tt::CBIndex::c_11;  // R x Sk table tiles per band chunk (reader)
    constexpr uint32_t cb_bias = tt::CBIndex::c_13; // Sk bias (+ mask) tiles per band chunk
    constexpr uint32_t cb_sink = tt::CBIndex::c_14; // Inkling-2: per-row sink / scale (every column), 1 tile
    constexpr uint32_t cb_out_accumulate_im_2 = tt::CBIndex::c_23;
    constexpr uint32_t cb_out_o = tt::CBIndex::c_16;
    constexpr uint32_t cb_out_m = tt::CBIndex::c_17;
    constexpr uint32_t cb_out_l = tt::CBIndex::c_18;
    constexpr uint32_t cb_out_final = tt::CBIndex::c_20;

    uint32_t arg_idx = 0;
    const uint32_t core_num_in_reduce = get_arg_val<uint32_t>(arg_idx++);
    const bool is_tree_root = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t parent_core_in_group = get_arg_val<uint32_t>(arg_idx++);
    const bool has_parent = parent_core_in_group != UINT32_MAX;
    uint32_t children_per_round[MAX_TREE_REDUCTION_ROUNDS];
    for (uint32_t r = 0; r < MAX_TREE_REDUCTION_ROUNDS; ++r) {
        children_per_round[r] = get_arg_val<uint32_t>(arg_idx++);
    }
    const uint32_t group = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t n_groups = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t n_units = get_arg_val<uint32_t>(arg_idx++);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q_in, cb_k_in, cb_qk_im);
    // Work units: unit = item * num_kv_heads + kv head. A unit is one 32-row Q tile (item = R consecutive rows of the
    // sequence x the G query heads of one kv head), computed by the num_cores_per_head cores of a group (decode: one
    // item, groups = kv heads, cores split the K span + tree reduction; prefill: one core per group, the items spread
    // over the groups round-robin, no reduction). Positions page `item` = that item's R rows; q / rq / out row r of
    // item i = sequence row i * R + r.
    for (uint32_t unit = group; unit < n_units; unit += n_groups) {
    // the rows' positions (mailbox-synchronized reads, stock op #27979): item m's R rows in page m
    uint32_t pos[MI * R];
    CircularBuffer(cb_cur_pos).wait_front(MI);
    for (uint32_t j = 0; j < MI * R; ++j) {
        pos[j] = read_tile_value(cb_cur_pos, j / R, j % R);
    }
    CircularBuffer(cb_cur_pos).pop_front(MI);
    uint32_t lim[MI * R];
    uint32_t max_pos = 0, min_lim = 0;
    if (!rows_sdpa_limits<MI * R>([&](uint32_t j) { return pos[j]; }, lim, max_pos, min_lim)) {
        continue;
    }

    uint32_t min_lo = max_pos, max_lo = 0;
    for (uint32_t j = 0; j < MI * R; ++j) {
        const uint32_t l = inkling_lo(lim[j], W);
        min_lo = l < min_lo ? l : min_lo;
        max_lo = l > max_lo ? l : max_lo;
    }
    const uint32_t Sk_chunk_t_dynamic = get_dynamic_Sk_chunk_t<Sk_chunk_t, max_dynamic_chunk_size>(max_pos);
    const uint32_t k_chunk_size_dynamic = Sk_chunk_t_dynamic * tt::constants::TILE_HEIGHT;
    const uint32_t W_eff = W == 0 ? 0 : max_pos - min_lo + 1;
    uint32_t cores_eff;
    auto [PSt, k_num_chunks, k_chunk_start, k_chunk_end, window_start_unaligned, window_start_chunk] =
        inkling_workload(max_pos, core_num_in_reduce, num_cores_per_head, k_chunk_size_dynamic, W_eff, cores_eff);
    if (k_chunk_start == k_chunk_end) {
        continue;
    }

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

    matmul_init(cb_q_in, cb_k_in);
    CircularBuffer(cb_q_in).wait_front(q_chunk_tiles);

    // dynamic matmul configuration (stock DYNAMIC_CHUNK_SIZE path; also exact for a fixed Sk <= dst)
    const uint32_t out_num_blocks_dynamic = 1;

    uint32_t cb_cur_max = cb_max_1;
    uint32_t cb_prev_max = cb_max_2;
    uint32_t cb_cur_sum = cb_sum_1;
    uint32_t cb_prev_sum = cb_sum_2;
    uint32_t cb_out_mm = cb_out_accumulate_im;

    {
    DeviceZoneScopedN("S_C_KV");
    for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
        reconfig_data_format(cb_k_in, cb_q_in);
        pack_reconfig_data_format(cb_qk_im);
        uint32_t tb, Skc;  // this chunk's key tiles (SDPA_TRIM: the last one only its live tiles)
        inkling_chunk_tiles(
            k_chunk, Sk_chunk_t_dynamic, max_pos, min_lo, k_chunk + 1 == k_chunk_end, tb, Skc, k_num_chunks,
            window_start_chunk, cores_eff);
        const bool masked = inkling_tiles_masked(tb, Skc, min_lim, W, max_lo);
        const bool band = inkling_tiles_band(tb, Skc, min_lim, E);
        const uint32_t out_in0_block_w_dynamic = Skc;
        const uint32_t qk_chunk_tiles_dynamic = Sq_chunk_t * Skc;

        {
        DeviceZoneScopedN("S_C_BIAS");
        if (band) {
            /* per item m: BIAS_m = sum_j RQ_{m,j} @ TBL_{m,j} (+ the writer's causal / window mask for its rows):
               [1 x R] @ [R x Sk] tiles (the reader's tables and the writer's masks arrive item by item) */
            reconfig_data_format(cb_tbl, cb_rq);
            pack_reconfig_data_format(cb_bias);
            for (uint32_t m = 0; m < MI; ++m) {
                inkling_matmul_rows(
                    cb_rq, cb_tbl, cb_bias, 1, Skc, R, false, masked, cb_mask_in, cb_zero_in, m * R, 0);
                if (masked) {
                    CircularBuffer(cb_mask_in).pop_front(Skc);
                }
            }
            CircularBuffer(cb_bias).wait_front(qk_chunk_tiles_dynamic);
            reconfig_data_format(cb_k_in, cb_q_in);
            pack_reconfig_data_format(cb_qk_im);
        }

        }
        {
        DeviceZoneScopedN("S_C_QK");
        /* QK = Q_CHUNK @ K_CHUNK (+ each Q tile row's bias block, added on DST) */
        inkling_matmul_rows(
            cb_q_in, cb_k_in, cb_qk_im, Sq_chunk_t, Skc, DHt, true, band, cb_bias, cb_zero_in, 0, Skc);
        if (band) {
            CircularBuffer(cb_bias).pop_front(qk_chunk_tiles_dynamic);
        }
        }
        {
        DeviceZoneScopedN("S_C_SM");
        reconfig_data_format(cb_qk_im, cb_identity_scale_in);
        pack_reconfig_data_format(cb_cur_max);
        reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, Sq_chunk_t, vector_mode>(
            cb_cur_max, cb_prev_max, Skc, k_chunk > k_chunk_start);
        reconfig_data_format(cb_qk_im, cb_cur_max);
        pack_reconfig_data_format(cb_qk_im);
        /* QK = exp((QK - cur_max) * scale) */
        sub_exp_block_bcast_cols_inplace<cb_qk_im, Sq_chunk_t, scale_fp32, true, false, vector_mode>(
            cb_cur_max, cb_cur_sum, Skc);
        CircularBuffer(cb_qk_im).wait_front(qk_chunk_tiles_dynamic);

        reconfig_data_format(cb_qk_im, cb_identity_scale_in);
        pack_reconfig_data_format(cb_cur_sum);
        reduce_c<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, Sq_chunk_t, vector_mode>(
            cb_cur_sum, cb_cur_sum, Skc, false);

        }
        {
        DeviceZoneScopedN("S_C_PV");
        /* OUT_IM = QK @ V_CHUNK */
        reconfig_data_format(cb_v_in, cb_qk_im);
        pack_reconfig_data_format(cb_out_im);
        matmul_blocks(
            cb_qk_im,
            cb_v_in,
            cb_out_mm,
            Sq_chunk_t,
            vDHt,
            Skc,
            out_num_blocks_dynamic,
            out_in0_num_subblocks,
            out_in1_num_subblocks,
            out_in0_block_w_dynamic,
            out_subblock_h,
            out_subblock_w,
            false,
            false,
            cb_mask_in,
            cb_zero_in);
        reconfig_data_format_srca(cb_out_im);
        CircularBuffer(cb_qk_im).pop_front(qk_chunk_tiles_dynamic);

        }
        if (k_chunk == k_chunk_start) {
            cb_out_mm = cb_out_im;
        } else {
            DeviceZoneScopedN("S_C_COR");
            reconfig_data_format(cb_prev_max, cb_cur_max);
            pack_reconfig_data_format(cb_exp_max_diff);
            /* EXP_MAX_DIFF = exp((PREV_MAX - CUR_MAX) * scale) */
            sub_exp_block<scale_fp32>(cb_prev_max, cb_cur_max, cb_exp_max_diff, Sq_chunk_t);
            CircularBuffer(cb_prev_max).pop_front(Sq_chunk_t);
            /* PREV_SUM *= EXP_MAX_DIFF */
            mul_block_inplace(cb_prev_sum, cb_exp_max_diff, Sq_chunk_t);
            /* OUT_ACC *= EXP_MAX_DIFF */
            reconfig_data_format(cb_out_accumulate_im, cb_exp_max_diff);
            pack_reconfig_data_format(cb_out_accumulate_im);
            mul_block_bcast_cols<Sq_chunk_t, vDHt, true, false>(
                cb_out_accumulate_im, cb_exp_max_diff, cb_out_accumulate_im);
            /* CUR_SUM += PREV_SUM */
            reconfig_data_format(cb_cur_sum, cb_prev_sum);
            pack_reconfig_data_format(cb_cur_sum);
            add_block_inplace<true>(cb_cur_sum, cb_prev_sum, Sq_chunk_t);
            /* OUT_ACC += OUT_IM */
            reconfig_data_format(cb_out_accumulate_im, cb_out_im);
            pack_reconfig_data_format(cb_out_accumulate_im);
            add_block_inplace<true>(cb_out_accumulate_im, cb_out_im, out_chunk_tiles);
        }
#if SDPA_FUSE
        swap_cbs(cb_cur_max, cb_prev_max);
        swap_cbs(cb_cur_sum, cb_prev_sum);
#else
        reconfig_data_format(cb_cur_max, cb_cur_max);
        pack_reconfig_data_format(cb_prev_max);
        move_block<true>(cb_cur_max, cb_prev_max, Sq_chunk_t);
        move_block<true>(cb_cur_sum, cb_prev_sum, Sq_chunk_t);
#endif
    }

    }
    // tree reduction (stock): combine each round's child into the local (O, max, sum) (decode only: MI > 1 units run
    // on one core each)
    {
    DeviceZoneScopedN("S_C_RED");
    if constexpr (MI == 1) {
    if (num_active_children > 0) {
        for (uint32_t round = 0; round < num_active_rounds; ++round) {
            if (active_children_per_round[round] == UINT32_MAX) {
                continue;
            }
            correction_block<scale_fp32, vector_mode>(
                cb_m_in,
                cb_l_in,
                cb_cur_max,
                cb_prev_max,
                cb_cur_sum,
                cb_prev_sum,
                cb_exp_max_diff,
                cb_exp_max_diff_2,
                Sq_chunk_t);
            mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(cb_out_accumulate_im, cb_exp_max_diff);
#if SDPA_FUSE
            acc_add_scaled<vDHt>(cb_out_accumulate_im, cb_out_o, cb_exp_max_diff_2);
            CircularBuffer(cb_prev_max).pop_front(Sq_chunk_t);
            CircularBuffer(cb_m_in).pop_front(Sq_chunk_t);
            swap_cbs(cb_cur_max, cb_prev_max);
            swap_cbs(cb_cur_sum, cb_prev_sum);
#else
            mul_block_bcast_cols<Sq_chunk_t, vDHt, false, false>(cb_out_o, cb_exp_max_diff_2, cb_out_accumulate_im_2);
            add_block_inplace<true>(cb_out_accumulate_im, cb_out_accumulate_im_2, out_chunk_tiles);
            CircularBuffer(cb_prev_max).pop_front(Sq_chunk_t);
            CircularBuffer(cb_m_in).pop_front(Sq_chunk_t);
            move_block<true>(cb_cur_max, cb_prev_max, Sq_chunk_t);
            move_block<true>(cb_cur_sum, cb_prev_sum, Sq_chunk_t);
#endif
        }
    }
    }

    }
    DeviceZoneScopedN("S_C_FIN");
    CircularBuffer(cb_sink).wait_front(Sq_chunk_t);
    if (is_tree_root) {
#if SDPA_FIN2
        /* Inkling-2 sink (a zero-value key with logit s per row): OUT = OUT_ACC / (SUM + exp((s - m) * scale)), the
           same as rescaling to max(m, s) first (exp((m - m') * scale) cancels) in 4 steps instead of 9 */
        reconfig_data_format(cb_sink, cb_prev_max);
        pack_reconfig_data_format(cb_exp_max_diff_2);
        sub_exp_block<scale_fp32>(cb_sink, cb_prev_max, cb_exp_max_diff_2, Sq_chunk_t);
        reconfig_data_format(cb_prev_sum, cb_exp_max_diff_2);
        pack_reconfig_data_format(cb_prev_sum);
        add_block_inplace<true>(cb_prev_sum, cb_exp_max_diff_2, Sq_chunk_t);
        reconfig_data_format(cb_prev_sum, cb_prev_sum);
        recip_block_inplace(cb_prev_sum, Sq_chunk_t);
        mul_block_bcast_cols<Sq_chunk_t, vDHt, false, false>(cb_out_accumulate_im, cb_prev_sum, cb_out_final);
        CircularBuffer(cb_prev_max).pop_front(Sq_chunk_t);
#else
        /* Inkling-2 sink (the stock op's attention-sink step with a per-row sink): m' = max(m, s);
           SUM = SUM * exp((m - m') * scale) + exp((s - m') * scale); OUT_ACC *= exp((m - m') * scale) */
        reconfig_data_format(cb_sink, cb_prev_max);
        pack_reconfig_data_format(cb_cur_max);
        max_block<vector_mode>(cb_sink, cb_prev_max, cb_cur_max, Sq_chunk_t);
#if SDPA_FUSE
        if constexpr (MI == 1) {
            /* EXP_MAX_DIFF, EXP_MAX_DIFF_2 in one DST round; then SUM = 1 / (SUM * EXP_MAX_DIFF + EXP_MAX_DIFF_2) and
               OUT_ACC *= EXP_MAX_DIFF in one; OUT = OUT_ACC * SUM */
            reconfig_data_format(cb_prev_max, cb_cur_max);
            pack_reconfig_data_format(cb_exp_max_diff);
            sub_exp_pair<scale_fp32>(cb_prev_max, cb_sink, cb_cur_max, cb_exp_max_diff, cb_exp_max_diff_2);
            CircularBuffer(cb_cur_max).pop_front(Sq_chunk_t);
            fin_sum_recip_scale<vDHt>(cb_prev_sum, cb_exp_max_diff, cb_exp_max_diff_2, cb_out_accumulate_im);
            mul_block_bcast_cols<Sq_chunk_t, vDHt, false, false>(cb_out_accumulate_im, cb_prev_sum, cb_out_final);
            CircularBuffer(cb_prev_max).pop_front(Sq_chunk_t);
        } else
#endif
        {
        reconfig_data_format(cb_prev_max, cb_cur_max);
        pack_reconfig_data_format(cb_exp_max_diff);
        sub_exp_block<scale_fp32>(cb_prev_max, cb_cur_max, cb_exp_max_diff, Sq_chunk_t);
        reconfig_data_format(cb_prev_sum, cb_exp_max_diff);
        pack_reconfig_data_format(cb_prev_sum);
        mul_block_inplace(cb_prev_sum, cb_exp_max_diff, Sq_chunk_t);
        reconfig_data_format(cb_sink, cb_cur_max);
        pack_reconfig_data_format(cb_exp_max_diff_2);
        sub_exp_block<scale_fp32>(cb_sink, cb_cur_max, cb_exp_max_diff_2, Sq_chunk_t);
        CircularBuffer(cb_cur_max).pop_front(Sq_chunk_t);
        reconfig_data_format(cb_prev_sum, cb_exp_max_diff_2);
        pack_reconfig_data_format(cb_prev_sum);
        add_block_inplace<true>(cb_prev_sum, cb_exp_max_diff_2, Sq_chunk_t);
        reconfig_data_format(cb_out_accumulate_im, cb_exp_max_diff);
        pack_reconfig_data_format(cb_out_accumulate_im);
        mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(cb_out_accumulate_im, cb_exp_max_diff);
        /* OUT = OUT_ACC * (1 / SUM) */
        reconfig_data_format(cb_prev_sum, cb_prev_sum);
        pack_reconfig_data_format(cb_prev_sum);
        recip_block_inplace(cb_prev_sum, Sq_chunk_t);
        mul_block_bcast_cols<Sq_chunk_t, vDHt, false, false>(cb_out_accumulate_im, cb_prev_sum, cb_out_final);
        CircularBuffer(cb_prev_max).pop_front(Sq_chunk_t);
        }
#endif
    } else if (has_parent) {
        if constexpr (MI == 1) {
#if SDPA_FUSE
            // the writer sends OUT_ACC / PREV_MAX / PREV_SUM in place once this (dataless) token lands: all packed
            CircularBuffer(cb_out_l).reserve_back(1);
            CircularBuffer(cb_out_l).push_back(1);
#else
            move_block<true>(cb_out_accumulate_im, cb_out_o, out_chunk_tiles);
            move_block<true>(cb_prev_max, cb_out_m, Sq_chunk_t);
            move_block<true>(cb_prev_sum, cb_out_l, Sq_chunk_t);
#endif
        }
    }
    CircularBuffer(cb_q_in).pop_front(q_chunk_tiles);
    CircularBuffer(cb_rq).pop_front(MI * R);
    CircularBuffer(cb_sink).pop_front(Sq_chunk_t);
    }  // units
}
