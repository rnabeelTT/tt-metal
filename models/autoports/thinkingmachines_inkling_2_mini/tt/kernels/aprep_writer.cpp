// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Decode attention prep (tt/aprep.py), writer: unit (b, type) -> its bf16 output tiles: q [1, B, NH, HD] (tiles
// b * HT + c), r [1, B, NH, 32] (b), k / v [1, B, 32, HD] (b * HT + c; the KV-write layout, may be height-sharded),
// sink [1, B, NH, 32] (b).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t r_addr = get_arg_val<uint32_t>(1);
    const uint32_t k_addr = get_arg_val<uint32_t>(2);
    const uint32_t v_addr = get_arg_val<uint32_t>(3);
    const uint32_t s_addr = get_arg_val<uint32_t>(4);
    const uint32_t start = get_arg_val<uint32_t>(5);
    const uint32_t step = get_arg_val<uint32_t>(6);
    const uint32_t total = get_arg_val<uint32_t>(7);
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t HT = get_compile_time_arg_val(1);
    constexpr uint32_t BF = 2048;
    constexpr auto q_args = TensorAccessorArgs<2>();
    constexpr auto r_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto kv_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    const auto aq = TensorAccessor(q_args, q_addr, BF);
    const auto ar = TensorAccessor(r_args, r_addr, BF);
    const auto as = TensorAccessor(r_args, s_addr, BF);
    const auto ak = TensorAccessor(kv_args, k_addr, BF);
    const auto av = TensorAccessor(kv_args, v_addr, BF);
    for (uint32_t u = start; u < total; u += step) {
        const uint32_t b = u / 5, type = u % 5;
        const uint32_t nt = (type == 1 || type == 4) ? 1 : HT;
        for (uint32_t c = 0; c < nt; ++c) {
            cb_wait_front(cb_out, 1);
            const uint32_t l1 = get_read_ptr(cb_out);
            uint64_t dst;
            if (type == 0) {
                dst = aq.get_noc_addr(b * HT + c);
            } else if (type == 1) {
                dst = ar.get_noc_addr(b);
            } else if (type == 2) {
                dst = ak.get_noc_addr(b * HT + c);
            } else if (type == 3) {
                dst = av.get_noc_addr(b * HT + c);
            } else {
                dst = as.get_noc_addr(b);
            }
            noc_async_write(l1, dst, BF);
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
    }
}
