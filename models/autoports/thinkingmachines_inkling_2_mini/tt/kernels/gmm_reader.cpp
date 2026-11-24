// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Grouped expert matmul (tt/gmm.py), reader: this core's column block b of the groups of experts [e0, e1) (the ids are
// sorted: each expert's weights are read by one slice only). Per group: its expert's
// [Kt x nb] weight block, pushed once per run of groups of one expert (the CB holds two blocks: the next expert's
// loads while the compute works on the current one), then the group's mt row tiles of A ([Kt] tiles each). Padding
// groups (id >= E) are skipped (by the compute and the writer too: they read the ids this pushes to cb_ids). The NB cores of a group slice form a rectangle: the b = 0
// core (a corner) reads each A row tile from DRAM and multicasts it to the rest of the rectangle (same CB slot on
// every core: they consume in lockstep); semaphore 0 (on the sender): receivers ready, 1 (on the receivers): the tile
// row landed.
//
// runtime args: b, e0, e1, sender x / y, multicast x0 y0 x1 y1 (NOC 0), G, s, c0
// common args: a, w, ids, ws, wo, wkt, k0, koff, rw
// (expert e's weights are w's expert e * ws + wo + s: a sub-shard of experts stacked as virtual experts,
// tt/moe_fused.py; this core's weight columns start at c0; A's columns at tile koff of its KA-tile rows)
// GMM_ACT: the block's nb = 2h columns are the gate columns [c0, c0 + h) and the up columns Wt + [c0, c0 + h), and
// each row tile's routing weights (column 0 of rw's tile, spread to every column) go to cb_rw.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_ids = get_compile_time_arg_val(2);
    constexpr uint32_t Kt = get_compile_time_arg_val(3);
    constexpr uint32_t Nt = get_compile_time_arg_val(4);
    constexpr uint32_t nb = get_compile_time_arg_val(5);
    constexpr uint32_t mt = get_compile_time_arg_val(6);
    constexpr uint32_t E = get_compile_time_arg_val(7);
    constexpr uint32_t ids_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t w_page = get_compile_time_arg_val(9);
    constexpr uint32_t KA = get_compile_time_arg_val(10);
    constexpr uint32_t cb_rw = get_compile_time_arg_val(11);
    constexpr uint32_t Wt = get_compile_time_arg_val(12);
    constexpr uint32_t NB = get_compile_time_arg_val(13);  // (cores per group slice: the A multicast's NB - 1 receivers)
    constexpr auto a_args = TensorAccessorArgs<14>();
    constexpr auto w_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto i_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto r_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    const uint32_t b = get_arg_val<uint32_t>(0), e0 = get_arg_val<uint32_t>(1), e1 = get_arg_val<uint32_t>(2);
    const uint32_t G = get_arg_val<uint32_t>(9);
    const uint32_t sub = get_arg_val<uint32_t>(10), c0 = get_arg_val<uint32_t>(11);
    const uint32_t sx = get_arg_val<uint32_t>(3), sy = get_arg_val<uint32_t>(4);
    const uint64_t mc_base = get_noc_multicast_addr(get_arg_val<uint32_t>(5), get_arg_val<uint32_t>(6),
                                                    get_arg_val<uint32_t>(7), get_arg_val<uint32_t>(8), 0);
    const uint32_t s_ready = get_semaphore(0), s_valid = get_semaphore(1);
    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s_ready);
    volatile tt_l1_ptr uint32_t* valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s_valid);
    if (b == 0) {
        *valid = 1;  // (the value the multicast copies to the receivers)
    }
    const auto aa = TensorAccessor(a_args, get_common_arg_val<uint32_t>(0), 2048);
    const auto aw = TensorAccessor(w_args, get_common_arg_val<uint32_t>(1), w_page);
    const auto ai = TensorAccessor(i_args, get_common_arg_val<uint32_t>(2), ids_bytes);
    const uint32_t ws = get_common_arg_val<uint32_t>(3), wo = get_common_arg_val<uint32_t>(4) + sub;
    // (K rows [k0, k0 + Kt) of each expert's wkt-row block: one K segment of the weights)
    const uint32_t wkt = get_common_arg_val<uint32_t>(5), k0 = get_common_arg_val<uint32_t>(6);
    const uint32_t koff = get_common_arg_val<uint32_t>(7);
#if GMM_ACT
    const auto ar = TensorAccessor(r_args, get_common_arg_val<uint32_t>(8), 2048);
#endif

    cb_reserve_back(cb_ids, 1);
    const uint32_t il = get_write_ptr(cb_ids);
    noc_async_read(ai.get_noc_addr(0), il, ids_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_ids, 1);
    volatile tt_l1_ptr uint16_t* ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(il);

    uint32_t prev = 0xFFFFFFFF;
    for (uint32_t g = 0; g < G; ++g) {
        const uint32_t e = ids[g];
        if (e < e0) {
            continue;
        }
        if (e >= e1) {
            break;  // (sorted; padding groups carry ids >= E >= e1)
        }
        if (e != prev) {
            cb_reserve_back(cb_w, Kt * nb);
            const uint32_t wl = get_write_ptr(cb_w);
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                for (uint32_t j = 0; j < nb; ++j) {
#if GMM_ACT
                    const uint32_t col = j < nb / 2 ? c0 + j : Wt + c0 + j - nb / 2;
#else
                    const uint32_t col = c0 + j;
#endif
                    noc_async_read(aw.get_noc_addr(((e * ws + wo) * wkt + k0 + kt) * Nt + col), wl + (kt * nb + j) * w_page,
                                   w_page);
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_w, Kt * nb);
            prev = e;
        }
        for (uint32_t mi = 0; mi < mt; ++mi) {
#if GMM_ACT
            cb_reserve_back(cb_rw, 1);
            const uint32_t rl = get_write_ptr(cb_rw);
            noc_async_read(ar.get_noc_addr(g * mt + mi), rl, 2048);
            noc_async_read_barrier();
            // column 0 of each row (faces 0 / 2) -> all 32 columns (tt/kernels/moeact_reader.cpp)
            volatile tt_l1_ptr uint16_t* t = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rl);
            for (uint32_t hf = 0; hf < 2; ++hf) {
                for (uint32_t row = 0; row < 16; ++row) {
                    const uint32_t base = hf * 512 + row * 16;
                    const uint16_t v = t[base];
                    for (uint32_t col = 0; col < 16; ++col) {
                        t[base + col] = v;
                        t[base + 256 + col] = v;
                    }
                }
            }
            cb_push_back(cb_rw, 1);
#endif
            cb_reserve_back(cb_a, Kt);
            const uint32_t al = get_write_ptr(cb_a);
            if (b == 0) {
                for (uint32_t kt = 0; kt < Kt; ++kt) {
                    noc_async_read(aa.get_noc_addr((g * mt + mi) * KA + koff + kt), al + kt * 2048, 2048);
                }
                noc_async_read_barrier();
                if constexpr (NB > 1) {
                    noc_semaphore_wait(ready, NB - 1);
                    noc_semaphore_set(ready, 0);
                    noc_async_write_multicast(al, mc_base | al, Kt * 2048, NB - 1, true);
                    noc_semaphore_set_multicast(s_valid, mc_base | s_valid, NB - 1);
                    noc_async_writes_flushed();  // (the slot is not refilled before the multicast has read it)
                }
            } else {
                noc_semaphore_set(valid, 0);
                noc_semaphore_inc(get_noc_addr(sx, sy, s_ready), 1);
                noc_semaphore_wait(valid, 1);
            }
            cb_push_back(cb_a, Kt);
        }
    }
    noc_async_write_barrier();  // (the A multicasts and semaphore updates)
    noc_async_atomic_barrier();
}
