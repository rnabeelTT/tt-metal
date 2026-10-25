// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Per head n: the updated state tiles back to the state's own pages (in place), then row 0 of the o tiles into row
// n % 32 of the [1,1,B*H,V] output (two 64-byte face-row writes per tile).
// KDA_ACTIVE_MODE: args 2 / 3 are this core's index and the core count; the active heads come in the reader's order
// (active head j -> core j % cores), then the inactive heads' output rows are zeroed in the same round-robin.
namespace {
constexpr uint32_t CB_OUT = 6, CB_ST = 7, CB_ACT = 30, CB_ZERO = 31;
constexpr uint32_t FACE_ROW_BYTES = 64;
constexpr uint32_t FACE_BYTES = 1024;
inline uint32_t row_offset(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + c) * FACE_BYTES + (r & 15) * FACE_ROW_BYTES;
}
}  // namespace

void kernel_main() {
    const uint32_t state_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t start = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);

    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr auto state_args = TensorAccessorArgs<2>();
    constexpr auto out_args = TensorAccessorArgs<state_args.next_compile_time_args_offset()>();
    constexpr uint32_t kv = Kt * Vt;

    const uint32_t page = get_local_cb_interface(CB_ST).fifo_page_size;
    const auto s_state = TensorAccessor(state_args, state_addr);
    const auto s_out = TensorAccessor(out_args, out_addr);
    Noc noc;
    CircularBuffer cst(CB_ST), co(CB_OUT);
#ifdef KDA_ACTIVE_MODE
    constexpr uint32_t H = get_compile_time_arg_val(out_args.next_compile_time_args_offset());
    constexpr auto act_args = TensorAccessorArgs<out_args.next_compile_time_args_offset() + 1>();
    const auto s_act = TensorAccessor(act_args, get_arg_val<uint32_t>(4));
    const uint32_t core_idx = start;
    const uint32_t num_cores = count;
    uint8_t act[KDA_BATCH], idle[KDA_BATCH];  // batch <= 256 (host check)
    uint32_t n_active = 0, n_idle = 0;
    {
        CircularBuffer ca(CB_ACT);
        // the second page: the reader reads the flags into the first
        noc.async_read(s_act, ca, KDA_ACT_BYTES, {.page_id = 0}, {.offset_bytes = page});
        noc.async_read_barrier();
        volatile tt_l1_ptr uint32_t* flags = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ca.get_write_ptr() + page);
        for (uint32_t b = 0; b < KDA_BATCH; ++b) {
            if (flags[b] != 0) {
                act[n_active++] = b;
            } else {
                idle[n_idle++] = b;
            }
        }
    }
    for (uint32_t j = core_idx; j < n_active * H; j += num_cores) {
        const uint32_t n = act[j / H] * H + (j - (j / H) * H);
#else
    for (uint32_t w = 0; w < count; ++w) {
        const uint32_t n = start + w;
#endif
        cst.wait_front(kv);
        for (uint32_t i = 0; i < kv; ++i) {
            noc.async_write(cst, s_state, page, {.offset_bytes = i * page}, {.page_id = n * kv + i});
        }
        co.wait_front(Vt);
        const uint32_t r = n & 31;
        for (uint32_t j = 0; j < Vt; ++j) {
            for (uint32_t c = 0; c < 2; ++c) {
                noc.async_write(
                    co,
                    s_out,
                    FACE_ROW_BYTES,
                    {.offset_bytes = j * 4 * FACE_BYTES + c * FACE_BYTES},
                    {.page_id = (n >> 5) * Vt + j, .offset_bytes = row_offset(r, c)});
            }
        }
        noc.async_write_barrier();
        cst.pop_front(kv);
        co.pop_front(Vt);
    }
#ifdef KDA_ACTIVE_MODE
    // inactive heads: zero output rows (their state is untouched)
    if (n_idle * H > core_idx) {
        CircularBuffer cz(CB_ZERO);
        noc.async_write_zeros(cz, FACE_BYTES);
        noc.write_zeros_l1_barrier();
        for (uint32_t j = core_idx; j < n_idle * H; j += num_cores) {
            const uint32_t n = idle[j / H] * H + (j - (j / H) * H);
            const uint32_t r = n & 31;
            for (uint32_t jj = 0; jj < Vt; ++jj) {
                for (uint32_t c = 0; c < 2; ++c) {
                    noc.async_write(
                        cz,
                        s_out,
                        FACE_ROW_BYTES,
                        {.offset_bytes = 0},
                        {.page_id = (n >> 5) * Vt + jj, .offset_bytes = row_offset(r, c)});
                }
            }
        }
        noc.async_write_barrier();
    }
#endif
}
