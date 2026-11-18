// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Expert-sorted MoE dispatch bookkeeping, one core, integer counting sort over the T*k (token, slot) pairs.
//   ids     [.., T, k] TILE uint16/uint32: expert of (token t, slot j); ids >= E are sentinels (ignored)
//   scores  [.., T, k] TILE bf16: routing weight of (t, j)
// Both are read straight from the top-k output: ceil(T/32) tiles (one tile column, k <= 32), indexed in place
// (32x32 tile = 4 16x16 faces: face (r/16)*2 + (c/16), element (r%16)*16 + c%16).
// Rows are laid out expert by expert (ascending id), each expert's rows padded to a multiple of `group`; the
// trailing rows up to Rd are the all-padding tail. Outputs:
//   src_tok[d]   source token of dispatched row d (0 for padding rows)
//   weights[d]   routing weight of row d (0 for padding rows)
//   group_ids[g] expert of group g (pad_group_id for padding groups: 0, or a sentinel >= E)
//   undispatch[i] destination row of pair i

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t rt = 0;
    const uint32_t ids_addr = get_arg_val<uint32_t>(rt++);
    const uint32_t sc_addr = get_arg_val<uint32_t>(rt++);
    const uint32_t src_addr = get_arg_val<uint32_t>(rt++);
    const uint32_t w_addr = get_arg_val<uint32_t>(rt++);
    const uint32_t gid_addr = get_arg_val<uint32_t>(rt++);
    const uint32_t und_addr = get_arg_val<uint32_t>(rt++);

    constexpr uint32_t R = get_compile_time_arg_val(0);
    (void)R;
    constexpr uint32_t T = get_compile_time_arg_val(1);
    constexpr uint32_t K = get_compile_time_arg_val(2);
    constexpr uint32_t E = get_compile_time_arg_val(3);
    constexpr uint32_t GS = get_compile_time_arg_val(4);
    constexpr uint32_t RD = get_compile_time_arg_val(5);
    constexpr uint32_t G = get_compile_time_arg_val(6);
    constexpr uint32_t ids_page = get_compile_time_arg_val(7);
    constexpr uint32_t sc_page = get_compile_time_arg_val(8);
    constexpr uint32_t src_page = get_compile_time_arg_val(9);
    constexpr uint32_t w_page = get_compile_time_arg_val(10);
    constexpr uint32_t gid_page = get_compile_time_arg_val(11);
    constexpr uint32_t und_page = get_compile_time_arg_val(12);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(13);
    constexpr uint32_t ids_elem_bytes = get_compile_time_arg_val(14);
    constexpr uint32_t pad_group_id = get_compile_time_arg_val(15);

    constexpr auto ids_args = TensorAccessorArgs<16>();
    constexpr auto sc_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto src_args = TensorAccessorArgs<sc_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto gid_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto und_args = TensorAccessorArgs<gid_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_ids = get_named_compile_time_arg_val("cb_ids");
    constexpr uint32_t cb_scores = get_named_compile_time_arg_val("cb_scores");
    constexpr uint32_t cb_src = get_named_compile_time_arg_val("cb_src");
    constexpr uint32_t cb_w = get_named_compile_time_arg_val("cb_w");
    constexpr uint32_t cb_gid = get_named_compile_time_arg_val("cb_gid");
    constexpr uint32_t cb_und = get_named_compile_time_arg_val("cb_und");
    constexpr uint32_t cb_scratch = get_named_compile_time_arg_val("cb_scratch");

    Noc noc;
    const auto s_ids = TensorAccessor(ids_args, ids_addr);
    const auto s_sc = TensorAccessor(sc_args, sc_addr);
    const auto s_src = TensorAccessor(src_args, src_addr);
    const auto s_w = TensorAccessor(w_args, w_addr);
    const auto s_gid = TensorAccessor(gid_args, gid_addr);
    const auto s_und = TensorAccessor(und_args, und_addr);

    const uint32_t ids_l1 = DataflowBuffer(cb_ids).get_write_ptr();
    const uint32_t sc_l1 = DataflowBuffer(cb_scores).get_write_ptr();
    const uint32_t src_l1 = DataflowBuffer(cb_src).get_write_ptr();
    const uint32_t w_l1 = DataflowBuffer(cb_w).get_write_ptr();
    const uint32_t gid_l1 = DataflowBuffer(cb_gid).get_write_ptr();
    const uint32_t und_l1 = DataflowBuffer(cb_und).get_write_ptr();
    const uint32_t scratch_l1 = DataflowBuffer(cb_scratch).get_write_ptr();

    // ---- inputs: every tile of the [T, k] top-k outputs
    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        noc.async_read(
            s_ids, CoreLocalMem<uint32_t>(ids_l1), ids_page, {.page_id = tile}, {.offset_bytes = tile * ids_page});
        noc.async_read(
            s_sc, CoreLocalMem<uint16_t>(sc_l1), sc_page, {.page_id = tile}, {.offset_bytes = tile * sc_page});
    }
    noc.async_read_barrier();

    // The inputs were written to L1 by the NOC: drop stale cache lines, then read them through plain pointers (the
    // scratch arrays are this core's alone; the outputs stay volatile, the NOC reads them back).
    invalidate_l1_cache();
    const uint16_t* ids16 = reinterpret_cast<const uint16_t*>(ids_l1);
    const uint32_t* ids32 = reinterpret_cast<const uint32_t*>(ids_l1);
    const uint16_t* scores = reinterpret_cast<const uint16_t*>(sc_l1);
    // (token t, slot j) inside the tiled buffers: tile t / 32, face (row / 16) * 2 + (j / 16), element (row % 16) * 16
    // + j % 16; a token's slots j < 16 are contiguous (k <= 16 for these models: one row of one face)
    auto row_base = [](uint32_t t) -> uint32_t {
        const uint32_t rr = t & 31;
        return (t >> 5) * 1024 + (rr >> 4) * 512 + (rr & 15) * 16;
    };
    auto slot_off = [](uint32_t j) -> uint32_t { return j < 16 ? j : 256 + (j - 16); };
    auto id_at = [&](uint32_t o) -> uint32_t {
        return ids_elem_bytes == 2 ? static_cast<uint32_t>(ids16[o]) : ids32[o];
    };
    // plain stores (a fence before the NOC reads the outputs back, below)
    uint32_t* src_tok = reinterpret_cast<uint32_t*>(src_l1);
    uint16_t* weights = reinterpret_cast<uint16_t*>(w_l1);
    uint16_t* group_ids = reinterpret_cast<uint16_t*>(gid_l1);
    uint32_t* undispatch = reinterpret_cast<uint32_t*>(und_l1);
    uint32_t* counts = reinterpret_cast<uint32_t*>(scratch_l1);
    uint32_t* offsets = counts + E;
    uint32_t* pos = offsets + E;  // next free row of each expert

    for (uint32_t e = 0; e < E; ++e) {
        counts[e] = 0;
    }
    constexpr bool word_ids = ids_elem_bytes == 2 && K <= 16 && K % 2 == 0;
    for (uint32_t t = 0; t < T; ++t) {
        const uint32_t b = row_base(t);
        if constexpr (word_ids) {
            const uint32_t* w = reinterpret_cast<const uint32_t*>(ids16 + b);  // b is even: 4-byte aligned
#pragma GCC unroll 8
            for (uint32_t j = 0; j < K / 2; ++j) {
                const uint32_t v = w[j];
                const uint32_t e0 = v & 0xFFFF, e1 = v >> 16;
                if (e0 < E) {
                    counts[e0] += 1;
                }
                if (e1 < E) {
                    counts[e1] += 1;
                }
            }
        } else {
            for (uint32_t j = 0; j < K; ++j) {
                const uint32_t e = id_at(b + slot_off(j));
                if (e < E) {
                    counts[e] += 1;
                }
            }
        }
    }
    // padded per-expert offsets, the expert of every group, and zeros in the padding rows only (the real rows are all
    // written by the scatter below): the same row contents as zeroing every row first
    uint32_t off = 0;
    for (uint32_t e = 0; e < E; ++e) {
        offsets[e] = off;
        pos[e] = off;
        const uint32_t groups_e = (counts[e] + GS - 1) / GS;
        const uint32_t g0 = off / GS;
        for (uint32_t g = 0; g < groups_e; ++g) {
            group_ids[g0 + g] = static_cast<uint16_t>(e);
        }
        for (uint32_t d = off + counts[e]; d < off + groups_e * GS; ++d) {
            src_tok[d] = 0;
            weights[d] = 0;
        }
        off += groups_e * GS;
    }
    for (uint32_t g = off / GS; g < G; ++g) {
        group_ids[g] = static_cast<uint16_t>(pad_group_id);
    }
    for (uint32_t d = off; d < RD; ++d) {
        src_tok[d] = 0;
        weights[d] = 0;
    }
    // scatter every real pair to its row, token-major (t, then j): the same row order as before
    uint32_t i = 0;
    for (uint32_t t = 0; t < T; ++t) {
        const uint32_t b = row_base(t);
        for (uint32_t j = 0; j < K; ++j, ++i) {
            const uint32_t o = b + slot_off(j);
            const uint32_t e = id_at(o);
            uint32_t d;
            if (e < E) {
                d = pos[e];
                pos[e] = d + 1;
                src_tok[d] = t;
                weights[d] = scores[o];
            } else {
                d = RD - GS;  // sentinel pairs land in the trailing padding group (never read back)
            }
            undispatch[i] = d;
        }
    }

    __asm__ volatile("fence" ::: "memory");  // every L1 store above lands before the NOC reads the outputs

    // ---- outputs (one stick each)
    noc.async_write(CoreLocalMem<uint32_t>(src_l1), s_src, src_page, {}, {.page_id = 0});
    noc.async_write(CoreLocalMem<uint16_t>(w_l1), s_w, w_page, {}, {.page_id = 0});
    noc.async_write(CoreLocalMem<uint16_t>(gid_l1), s_gid, gid_page, {}, {.page_id = 0});
    noc.async_write(CoreLocalMem<uint32_t>(und_l1), s_und, und_page, {}, {.page_id = 0});
    noc.async_write_barrier();
}
