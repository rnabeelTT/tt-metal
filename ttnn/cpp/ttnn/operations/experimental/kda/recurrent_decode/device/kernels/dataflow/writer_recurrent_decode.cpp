// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Per head n: the updated state tiles back to the state's own pages (in place), then row 0 of the o tiles into row
// n % 32 of the [1,1,B*H,V] output (two 64-byte face-row writes per tile).
namespace {
constexpr uint32_t CB_OUT = 6, CB_ST = 7;
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
    for (uint32_t w = 0; w < count; ++w) {
        const uint32_t n = start + w;
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
}
