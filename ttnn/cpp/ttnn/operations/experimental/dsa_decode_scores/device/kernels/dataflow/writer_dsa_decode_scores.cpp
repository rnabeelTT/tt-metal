// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Per unit (s, j): row 0 of the computed score tile (active) or -inf (inactive) into row s % 32 of output tile
// (s / 32, j): two 64-byte face-row writes.
namespace {
constexpr uint32_t CB_NEGINF = 7, CB_WLIM = 8, CB_OUT = 16;
constexpr uint32_t FACE_ROW_BYTES = 64, FACE_BYTES = 1024;
inline uint32_t row_offset(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + c) * FACE_BYTES + (r & 15) * FACE_ROW_BYTES;
}
}  // namespace

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t lim_addr = get_arg_val<uint32_t>(1);
    const uint32_t start = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);

    constexpr uint32_t nblk = get_compile_time_arg_val(0);
    constexpr uint32_t lim_bytes = get_compile_time_arg_val(1);
    constexpr auto out_args = TensorAccessorArgs<2>();
    constexpr auto lim_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    if (count == 0) {
        return;
    }
    const auto s_out = TensorAccessor(out_args, out_addr);
    const auto s_lim = TensorAccessor(lim_args, lim_addr);
    Noc noc;
    CircularBuffer co(CB_OUT), cneg(CB_NEGINF), clim(CB_WLIM);
    // the writer's own copy of the limits (the active rule j * 32 < limits[s] mirrors the reader's)
    clim.reserve_back(1);
    noc.async_read(s_lim, clim, lim_bytes, {.page_id = 0}, {.offset_bytes = 0});
    cneg.reserve_back(1);
    volatile tt_l1_ptr uint32_t* neg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cneg.get_write_ptr());
    for (uint32_t i = 0; i < 16; ++i) {
        neg[i] = 0xFF800000u;  // fp32 -inf
    }
    noc.async_read_barrier();
    volatile tt_l1_ptr uint32_t* lim = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(clim.get_write_ptr());
    for (uint32_t w = 0; w < count; ++w) {
        const uint32_t u = start + w;
        const uint32_t s = u / nblk;
        const uint32_t j = u - s * nblk;
        const uint32_t page = (s >> 5) * nblk + j;
        const uint32_t r = s & 31;
        const bool active = j * 32 < lim[s];
        if (active) {
            co.wait_front(1);
            for (uint32_t c = 0; c < 2; ++c) {
                noc.async_write(
                    co,
                    s_out,
                    FACE_ROW_BYTES,
                    {.offset_bytes = c * FACE_BYTES},
                    {.page_id = page, .offset_bytes = row_offset(r, c)});
            }
            noc.async_write_barrier();
            co.pop_front(1);
        } else {
            for (uint32_t c = 0; c < 2; ++c) {
                noc.async_write(
                    cneg,
                    s_out,
                    FACE_ROW_BYTES,
                    {.offset_bytes = 0},
                    {.page_id = page, .offset_bytes = row_offset(r, c)});
            }
            noc.async_write_barrier();
        }
    }
}
