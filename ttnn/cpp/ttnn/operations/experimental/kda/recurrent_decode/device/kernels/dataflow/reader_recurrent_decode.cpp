// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Per head n = b * H + h of this core's run: the head's row of q, k, v, decay (row b of the packed [1,1,R,C] inputs) is
// gathered into row 0 of zeroed tiles, two 64-byte face-row reads per tile (fp32: face row = 16 x 4 B, 64-byte aligned
// in DRAM); beta into element [0,0]; then the head's Kt x Vt state tiles.
namespace {
constexpr uint32_t CB_Q = 0, CB_K = 1, CB_V = 2, CB_DECAY = 3, CB_BETA = 4, CB_S = 5, CB_SCRATCH = 17;
constexpr uint32_t FACE_ROW_BYTES = 64;  // 16 fp32
constexpr uint32_t FACE_BYTES = 1024;    // 16 x 16 fp32

// byte offset of row r (0..31), column half c (0: cols 0-15, 1: cols 16-31) inside an fp32 tile
inline uint32_t row_offset(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + c) * FACE_BYTES + (r & 15) * FACE_ROW_BYTES;
}

template <typename Accessor>
inline void gather_row(
    const Noc& noc, const Accessor& src, CircularBuffer& cb, uint32_t page0, uint32_t ntiles, uint32_t r) {
    for (uint32_t i = 0; i < ntiles; ++i) {
        for (uint32_t c = 0; c < 2; ++c) {
            noc.async_read(
                src,
                cb,
                FACE_ROW_BYTES,
                {.page_id = page0 + i, .offset_bytes = row_offset(r, c)},
                {.offset_bytes = i * 4 * FACE_BYTES + c * FACE_BYTES});
        }
    }
}

// zero every page of a CB (once per launch: the gathers only ever write row 0, so rows 1..31 stay zero)
inline void zero_cb(const Noc& noc, uint32_t cb_id, uint32_t pages, uint32_t page_bytes) {
    CircularBuffer cb(cb_id);
    noc.async_write_zeros(cb, pages * page_bytes);
}
}  // namespace

void kernel_main() {
    const uint32_t state_addr = get_arg_val<uint32_t>(0);
    const uint32_t qkv_addr = get_arg_val<uint32_t>(1);
    const uint32_t decay_addr = get_arg_val<uint32_t>(2);
    const uint32_t beta_addr = get_arg_val<uint32_t>(3);
    const uint32_t start = get_arg_val<uint32_t>(4);
    const uint32_t count = get_arg_val<uint32_t>(5);

    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t H = get_compile_time_arg_val(2);
    constexpr uint32_t qkv_ct = get_compile_time_arg_val(3);
    constexpr uint32_t decay_ct = get_compile_time_arg_val(4);
    constexpr uint32_t beta_ct = get_compile_time_arg_val(5);
    constexpr auto state_args = TensorAccessorArgs<6>();
    constexpr auto qkv_args = TensorAccessorArgs<state_args.next_compile_time_args_offset()>();
    constexpr auto decay_args = TensorAccessorArgs<qkv_args.next_compile_time_args_offset()>();
    constexpr auto beta_args = TensorAccessorArgs<decay_args.next_compile_time_args_offset()>();
    constexpr uint32_t kv = Kt * Vt;
    if (count == 0) {
        return;
    }

    const uint32_t page = get_local_cb_interface(CB_S).fifo_page_size;
    const auto s_state = TensorAccessor(state_args, state_addr);
    const auto s_qkv = TensorAccessor(qkv_args, qkv_addr);
    const auto s_decay = TensorAccessor(decay_args, decay_addr);
    const auto s_beta = TensorAccessor(beta_args, beta_addr);

    Noc noc;
    CircularBuffer cq(CB_Q), ck(CB_K), cv(CB_V), cd(CB_DECAY), cb(CB_BETA), cs(CB_S), csc(CB_SCRATCH);
    zero_cb(noc, CB_Q, 2 * Kt, page);
    zero_cb(noc, CB_K, 2 * Kt, page);
    zero_cb(noc, CB_V, 2 * Vt, page);
    zero_cb(noc, CB_DECAY, 2 * Kt, page);
    zero_cb(noc, CB_BETA, 2, page);
    noc.write_zeros_l1_barrier();

    for (uint32_t w = 0; w < count; ++w) {
        const uint32_t n = start + w;
        const uint32_t b = n / H;
        const uint32_t h = n - b * H;
        const uint32_t tr = b >> 5;
        const uint32_t r = b & 31;

        cq.reserve_back(Kt);
        ck.reserve_back(Kt);
        cv.reserve_back(Vt);
        cd.reserve_back(Kt);
        cb.reserve_back(1);
        gather_row(noc, s_qkv, cq, tr * qkv_ct + h * Kt, Kt, r);
        gather_row(noc, s_qkv, ck, tr * qkv_ct + (H + h) * Kt, Kt, r);
        gather_row(noc, s_qkv, cv, tr * qkv_ct + 2 * H * Kt + h * Vt, Vt, r);
        gather_row(noc, s_decay, cd, tr * decay_ct + h * Kt, Kt, r);
        // beta: the 64-byte face row holding column h of row b, then the element into [0,0]
        const uint32_t c = h & 31;
        noc.async_read(
            s_beta,
            csc,
            FACE_ROW_BYTES,
            {.page_id = tr * beta_ct + (h >> 5), .offset_bytes = row_offset(r, c >> 4)},
            {.offset_bytes = 0});
        noc.async_read_barrier();
        volatile tt_l1_ptr uint32_t* scratch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(csc.get_write_ptr());
        volatile tt_l1_ptr uint32_t* beta_tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr());
        beta_tile[0] = scratch[c & 15];
        cq.push_back(Kt);
        ck.push_back(Kt);
        cv.push_back(Vt);
        cd.push_back(Kt);
        cb.push_back(1);

        cs.reserve_back(kv);
        for (uint32_t i = 0; i < kv; ++i) {
            noc.async_read(s_state, cs, page, {.page_id = n * kv + i}, {.offset_bytes = i * page});
        }
        noc.async_read_barrier();
        cs.push_back(kv);
    }
}
