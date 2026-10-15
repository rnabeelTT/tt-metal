// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Units u = s * num_blocks + j of this core's run. On a slot change: the slot's page-table row, q tiles and weights
// row (row s % 32 of the weights tile, two 64-byte face-row reads into row 0 of a zeroed tile). Per unit: the active
// flag (j * 32 < limits[s]) to compute, and for an active unit the key block's tiles through the page table.
namespace {
constexpr uint32_t CB_Q = 0, CB_W = 1, CB_K = 2, CB_CTRL = 3, CB_PT = 4, CB_LIM = 5;
constexpr uint32_t FACE_ROW_BYTES = 64, FACE_BYTES = 1024;
inline uint32_t row_offset(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + c) * FACE_BYTES + (r & 15) * FACE_ROW_BYTES;
}
}  // namespace

void kernel_main() {
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t k_addr = get_arg_val<uint32_t>(2);
    const uint32_t pt_addr = get_arg_val<uint32_t>(3);
    const uint32_t lim_addr = get_arg_val<uint32_t>(4);
    const uint32_t start = get_arg_val<uint32_t>(5);
    const uint32_t count = get_arg_val<uint32_t>(6);

    constexpr uint32_t Dt = get_compile_time_arg_val(0);
    constexpr uint32_t nblk = get_compile_time_arg_val(1);
    constexpr uint32_t pt_row_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t lim_bytes = get_compile_time_arg_val(3);
    constexpr auto q_args = TensorAccessorArgs<4>();
    constexpr auto w_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto k_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto pt_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto lim_args = TensorAccessorArgs<pt_args.next_compile_time_args_offset()>();
    if (count == 0) {
        return;
    }

    const uint32_t qk_page = get_local_cb_interface(CB_Q).fifo_page_size;
    const uint32_t w_page = get_local_cb_interface(CB_W).fifo_page_size;
    const auto s_q = TensorAccessor(q_args, q_addr);
    const auto s_w = TensorAccessor(w_args, w_addr);
    const auto s_k = TensorAccessor(k_args, k_addr);
    const auto s_pt = TensorAccessor(pt_args, pt_addr);
    const auto s_lim = TensorAccessor(lim_args, lim_addr);

    Noc noc;
    CircularBuffer cq(CB_Q), cw(CB_W), ck(CB_K), cc(CB_CTRL), cpt(CB_PT), clim(CB_LIM);
    noc.async_write_zeros(cw, 2 * w_page);  // weights rows land in row 0 only; rows 1..31 stay zero
    noc.write_zeros_l1_barrier();
    clim.reserve_back(1);
    noc.async_read(s_lim, clim, lim_bytes, {.page_id = 0}, {.offset_bytes = 0});
    cpt.reserve_back(1);
    noc.async_read_barrier();
    volatile tt_l1_ptr uint32_t* lim = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(clim.get_write_ptr());
    volatile tt_l1_ptr uint32_t* pt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cpt.get_write_ptr());

    uint32_t cur = 0xFFFFFFFFu;
    uint32_t limit = 0;
    for (uint32_t w = 0; w < count; ++w) {
        const uint32_t u = start + w;
        const uint32_t s = u / nblk;
        const uint32_t j = u - s * nblk;
        if (s != cur) {
            cur = s;
            limit = lim[s];
            noc.async_read(s_pt, cpt, pt_row_bytes, {.page_id = s}, {.offset_bytes = 0});
            cq.reserve_back(Dt);
            for (uint32_t d = 0; d < Dt; ++d) {
                noc.async_read(s_q, cq, qk_page, {.page_id = s * Dt + d}, {.offset_bytes = d * qk_page});
            }
            cw.reserve_back(1);
            for (uint32_t c = 0; c < 2; ++c) {
                noc.async_read(
                    s_w,
                    cw,
                    FACE_ROW_BYTES,
                    {.page_id = s >> 5, .offset_bytes = row_offset(s & 31, c)},
                    {.offset_bytes = c * FACE_BYTES});
            }
            noc.async_read_barrier();
            cq.push_back(Dt);
            cw.push_back(1);
        }
        const uint32_t active = (j * 32 < limit) ? 1u : 0u;
        cc.reserve_back(1);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cc.get_write_ptr())[0] = active;
        cc.push_back(1);
        if (active) {
            const uint32_t phys = pt[j];
            ck.reserve_back(Dt);
            for (uint32_t d = 0; d < Dt; ++d) {
                noc.async_read(s_k, ck, qk_page, {.page_id = phys * Dt + d}, {.offset_bytes = d * qk_page});
            }
            noc.async_read_barrier();
            ck.push_back(Dt);
        }
    }
}
