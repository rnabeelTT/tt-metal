// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Permutation tiles for the exact Sinkhorn row / column sums (see the compute kernel): tile 0 is the identity, tiles
// 1..n-1 shift each entry (i, j) of the flattened n x n matrix to (i, j+s mod n) (row sums), tiles n..2n-2 to
// (i+s mod n, j) (column sums). m @ P_s[q, p] = m[sigma_s(p)] with P_s[sigma_s(p), p] = 1: one nonzero product per
// output, which the FPU sums exactly. Pad columns p >= n*n map to themselves in the identity only.
inline void make_sinkhorn_perms(uint32_t cb_perm, uint32_t n) {
    CircularBuffer cb(cb_perm);
    const uint32_t tiles = 2 * n - 1;
    cb.reserve_back(tiles);
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr());
    for (uint32_t i = 0; i < tiles * 1024; ++i) {
        p[i] = 0;
    }
    auto set_one = [&](uint32_t t, uint32_t q, uint32_t col) {  // fp32 tile layout: 4 faces of 16 x 16
        const uint32_t face = (q >> 4) * 2 + (col >> 4);
        p[t * 1024 + face * 256 + (q & 15) * 16 + (col & 15)] = 0x3f800000u;  // 1.0f
    };
    const uint32_t nn = n * n;
    for (uint32_t col = 0; col < 32; ++col) {
        set_one(0, col, col);
        if (col >= nn) {
            continue;
        }
        const uint32_t i = col / n, j = col - (col / n) * n;
        for (uint32_t s = 1; s < n; ++s) {
            set_one(s, i * n + (j + s) % n, col);
            set_one(n - 1 + s, ((i + s) % n) * n + j, col);
        }
    }
    cb.push_back(tiles);
}

// Loads the 8 host-prepared constant tiles once (persistent), then streams the mixes
// token-tiles. See the program factory for the CB layout.
void kernel_main() {
    const uint32_t mixes_addr = get_arg_val<uint32_t>(0);
    const uint32_t consts_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_token_tiles = get_arg_val<uint32_t>(2);
    const uint32_t start_tile = get_arg_val<uint32_t>(3);  // this core's first mixes page

    constexpr uint32_t cb_mixes = get_compile_time_arg_val(0);
    constexpr uint32_t cb_consts = get_compile_time_arg_val(1);
    constexpr uint32_t cb_perm = get_compile_time_arg_val(2);
    constexpr uint32_t n = get_compile_time_arg_val(3);
    constexpr auto mixes_args = TensorAccessorArgs<4>();
    constexpr auto consts_args = TensorAccessorArgs<mixes_args.next_compile_time_args_offset()>();

    constexpr uint32_t num_const_tiles = 8;

    const uint32_t mixes_page = get_local_cb_interface(cb_mixes).fifo_page_size;
    const uint32_t consts_page = get_local_cb_interface(cb_consts).fifo_page_size;

    const auto s_mixes = TensorAccessor(mixes_args, mixes_addr);
    const auto s_consts = TensorAccessor(consts_args, consts_addr);

    Noc noc;
    CircularBuffer cb_c(cb_consts);
    CircularBuffer cb_m(cb_mixes);

    // Constants: load once, never popped by compute -> resident for every token-tile.
    for (uint32_t i = 0; i < num_const_tiles; ++i) {
        cb_c.reserve_back(1);
        noc.async_read(s_consts, cb_c, consts_page, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_c.push_back(1);
    }

    make_sinkhorn_perms(cb_perm, n);

    for (uint32_t t = 0; t < num_token_tiles; ++t) {
        cb_m.reserve_back(1);
        noc.async_read(s_mixes, cb_m, mixes_page, {.page_id = start_tile + t}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_m.push_back(1);
    }
}
