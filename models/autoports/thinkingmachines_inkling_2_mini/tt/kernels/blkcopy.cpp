// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Page-range copies between interleaved DRAM tensors (tt/blkcopy.py): each job copies n pages of page_size bytes from
// page sp of the tensor at src to page dp of the tensor at dst (the prefix-cache snapshots: sliding-window KV blocks
// and conv states into / out of their snapshot pools). Pages go through an L1 staging buffer, `per` pages per batch.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t buf_bytes = get_compile_time_arg_val(1);
    constexpr auto args = TensorAccessorArgs<2>();
    const uint32_t n_jobs = get_arg_val<uint32_t>(0);
    const uint32_t l1 = get_write_ptr(cb);
    uint32_t ai = 1;
    for (uint32_t j = 0; j < n_jobs; ++j) {
        const uint32_t src = get_arg_val<uint32_t>(ai);
        const uint32_t sp = get_arg_val<uint32_t>(ai + 1);
        const uint32_t dst = get_arg_val<uint32_t>(ai + 2);
        const uint32_t dp = get_arg_val<uint32_t>(ai + 3);
        const uint32_t n = get_arg_val<uint32_t>(ai + 4);
        const uint32_t ps = get_arg_val<uint32_t>(ai + 5);
        ai += 6;
        const auto as = TensorAccessor(args, src, ps);
        const auto ad = TensorAccessor(args, dst, ps);
        const uint32_t per = buf_bytes / ps;
        for (uint32_t p = 0; p < n; p += per) {
            const uint32_t k = (n - p) < per ? (n - p) : per;
            for (uint32_t i = 0; i < k; ++i) {
                noc_async_read(as.get_noc_addr(sp + p + i), l1 + i * ps, ps);
            }
            noc_async_read_barrier();
            for (uint32_t i = 0; i < k; ++i) {
                noc_async_write(l1 + i * ps, ad.get_noc_addr(dp + p + i), ps);
            }
            noc_async_write_barrier();
        }
    }
}
