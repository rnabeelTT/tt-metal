// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode router top-k + softmax for one token (moe_decode/topk.py RouterTopK), one data-movement core.
//
// 1. Reads row 0 of the [32, E] BF16 router logits (TILE layout, any interleaved / sharded placement): per 32x32 tile,
//    16 values at face 0 row 0 (byte 0) and 16 at face 1 row 0 (byte 512).
// 2. Top-k + softmax over the selected logits (router_topk.hpp).
// 3. Writes the k expert ids as one UINT16 row-major page ([1, 1, 1, k]) and the k weights as row 0 of one BF16
//    32x32 tile ([1, 1, 1, k] TILE; the rest of the tile zero).
//
// runtime args: [logits_addr, ids_addr, scores_addr]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "router_topk.hpp"

void kernel_main() {
    const uint32_t logits_addr = get_arg_val<uint32_t>(0);
    const uint32_t ids_addr = get_arg_val<uint32_t>(1);
    const uint32_t scores_addr = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_scr = get_compile_time_arg_val(0);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(1);  // E / 32
    constexpr uint32_t topk = get_compile_time_arg_val(2);
    constexpr uint32_t ids_page_bytes = get_compile_time_arg_val(3);
    constexpr auto logits_args = TensorAccessorArgs<4>();
    constexpr auto ids_args = TensorAccessorArgs<logits_args.next_compile_time_args_offset()>();
    constexpr auto scores_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();

    constexpr uint32_t n = n_tiles * 32;
    constexpr uint32_t tile_bytes = 2048;
    static_assert(topk <= 16, "the weights fill face 0 row 0 of the score tile");

    const auto s_logits = TensorAccessor(logits_args, logits_addr, tile_bytes);
    const auto s_ids = TensorAccessor(ids_args, ids_addr, ids_page_bytes);
    const auto s_scores = TensorAccessor(scores_args, scores_addr, tile_bytes);

    // scratch: [score tile 2048 B][logits 2 n B][ids page]
    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    const uint32_t score_l1 = scr;
    const uint32_t logits_l1 = scr + tile_bytes;
    const uint32_t ids_l1 = logits_l1 + 2 * n;

    for (uint32_t t = 0; t < n_tiles; ++t) {
        noc_async_read(s_logits.get_noc_addr(t, 0), logits_l1 + t * 64, 32);
        noc_async_read(s_logits.get_noc_addr(t, 512), logits_l1 + t * 64 + 32, 32);
    }
    volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(score_l1);
    for (uint32_t i = 0; i < tile_bytes / 4; ++i) {
        z[i] = 0;
    }
    noc_async_read_barrier();

    uint32_t ids[topk];
    uint16_t w[topk];
    invalidate_l1_cache();
    moe_topk::topk_softmax<topk, n>(reinterpret_cast<const uint32_t*>(logits_l1), ids, w);

    volatile tt_l1_ptr uint16_t* id_out = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ids_l1);
    volatile tt_l1_ptr uint16_t* w_out = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(score_l1);
    for (uint32_t j = 0; j < topk; ++j) {
        id_out[j] = static_cast<uint16_t>(ids[j]);
        w_out[j] = w[j];
    }
    noc_async_write(ids_l1, s_ids.get_noc_addr(0), ids_page_bytes);
    noc_async_write(score_l1, s_scores.get_noc_addr(0), tile_bytes);
    noc_async_write_barrier();
}
