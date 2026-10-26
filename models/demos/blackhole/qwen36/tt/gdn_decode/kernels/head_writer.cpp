// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// GDN decode head step, writer: the unit's new state tiles back in place ([Bmax, Nv, Dk, Dv] bf16 TILE, DRAM) and its
// gated output row into row b of the out-proj input ([1, B, Nv * Dv] bf16 TILE, L1): two 32-byte face-row writes per
// tile, disjoint from every other unit's (no read-modify-write).
//
// VSPLIT = P (one work item per core, item n = p * U + u; runtime args 4.. : U, then the NOC x, y of the P cores of
// this unit in part order): this core writes its Vt-tile slice of the state and output, and exchanges the gated
// RMSNorm's sum of o^2 with the unit's other parts: once every part has started (program semaphore 0, incremented by
// each peer at its start: a peer's cb_xs is not written before its program runs) it writes its FP32 partial into slot
// p of every part's cb_xs and increments their semaphore 1; once all P partials are in, the sum in part order (the same
// on every part) goes back to the compute as element 0 of a cb_nt tile.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t state_addr = get_arg_val<uint32_t>(0);
    const uint32_t y_addr = get_arg_val<uint32_t>(1);
    const uint32_t start = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);

    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t Nv = get_compile_time_arg_val(2);
    constexpr uint32_t cb_state = get_compile_time_arg_val(3);
    constexpr uint32_t cb_y = get_compile_time_arg_val(4);
    constexpr auto state_args = TensorAccessorArgs<5>();
    constexpr auto y_args = TensorAccessorArgs<state_args.next_compile_time_args_offset()>();
    constexpr uint32_t kv = Kt * Vt;
    constexpr uint32_t BF_TILE = 2048, BF_FACE = 512, BF_ROW = 32;

    const auto s_state = TensorAccessor(state_args, state_addr, BF_TILE);
    const auto s_y = TensorAccessor(y_args, y_addr, BF_TILE);

#ifdef VSPLIT
    constexpr uint32_t P = VSPLIT;
    constexpr uint32_t VTF = VT_FULL;
    constexpr uint32_t cb_nx = 32, cb_nt = 33, cb_xs = 34;
    const uint32_t U = get_arg_val<uint32_t>(4);
    auto peer_x = [](uint32_t q) { return get_arg_val<uint32_t>(5 + 2 * q); };
    auto peer_y = [](uint32_t q) { return get_arg_val<uint32_t>(6 + 2 * q); };
    const uint32_t ready_sem = get_semaphore(0);
    const uint32_t data_sem = get_semaphore(1);
    for (uint32_t q = 0; q < P; ++q) {  // "started" to every part (self included: one count per part)
        noc_semaphore_inc(get_noc_addr(peer_x(q), peer_y(q), ready_sem), 1);
    }
    const uint32_t xs = get_write_ptr(cb_xs);
#else
    constexpr uint32_t VTF = Vt;
    const uint32_t U = 0;
#endif

    for (uint32_t w = 0; w < count; ++w) {
#ifdef VSPLIT
        const uint32_t n = (start + w) % U;
        const uint32_t p = (start + w) / U;
#else
        const uint32_t n = start + w;
        const uint32_t p = 0;
#endif
        const uint32_t b = n / Nv;
        const uint32_t h = n - b * Nv;
#ifdef VSPLIT
        {
            cb_wait_front(cb_nx, 1);
            invalidate_l1_cache();
            const uint32_t mine = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_nx));
            volatile tt_l1_ptr uint32_t* own = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xs + 16 * P);
            own[0] = mine;  // the source of the slot writes (16-byte aligned)
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_sem), P);
            for (uint32_t q = 0; q < P; ++q) {
                noc_async_write(xs + 16 * P, get_noc_addr(peer_x(q), peer_y(q), xs + 16 * p), 4);
            }
            noc_async_write_barrier();
            for (uint32_t q = 0; q < P; ++q) {
                noc_semaphore_inc(get_noc_addr(peer_x(q), peer_y(q), data_sem), 1);
            }
            cb_pop_front(cb_nx, 1);
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_sem), P * (w + 1));
            invalidate_l1_cache();
            float total = 0.0f;
            for (uint32_t q = 0; q < P; ++q) {
                total += *reinterpret_cast<volatile tt_l1_ptr float*>(xs + 16 * q);
            }
            cb_reserve_back(cb_nt, 1);
            *reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(cb_nt)) = total;
            cb_push_back(cb_nt, 1);
        }
#endif
        cb_wait_front(cb_state, kv);
        const uint32_t s = get_read_ptr(cb_state);
        for (uint32_t ki = 0; ki < Kt; ++ki) {
            for (uint32_t j = 0; j < Vt; ++j) {
                const uint32_t dst = n * (Kt * VTF) + ki * VTF + p * Vt + j;
                noc_async_write(s + (ki * Vt + j) * BF_TILE, s_state.get_noc_addr(dst), BF_TILE);
            }
        }
        cb_wait_front(cb_y, Vt);
        const uint32_t y = get_read_ptr(cb_y);
        const uint32_t r = b & 31;
        for (uint32_t j = 0; j < Vt; ++j) {
            const uint32_t page = (b >> 5) * (Nv * VTF) + h * VTF + p * Vt + j;
            for (uint32_t c = 0; c < 2; ++c) {
                const uint32_t off = ((r >> 4) * 2 + c) * BF_FACE + (r & 15) * BF_ROW;
                noc_async_write(y + j * BF_TILE + c * BF_FACE, s_y.get_noc_addr(page, off), BF_ROW);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_state, kv);
        cb_pop_front(cb_y, Vt);
    }
#ifdef VSPLIT
    noc_async_atomic_barrier();
#endif
    (void)U;
}
