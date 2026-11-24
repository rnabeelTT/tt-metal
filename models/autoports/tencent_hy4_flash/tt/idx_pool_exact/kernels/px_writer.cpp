// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Indexer pooling (exact), writer (RISCV_0): unit u = (k, ct) -> page k 4 + ct of the output from cb 16.
// ct: [PAGE, acc(out)]; rt: [out addr, u0, nu]
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t PAGE = get_compile_time_arg_val(0);
    constexpr auto o_args = TensorAccessorArgs<1>();
    const uint32_t o_addr = get_arg_val<uint32_t>(0);
    const uint32_t u0 = get_arg_val<uint32_t>(1), nu = get_arg_val<uint32_t>(2);
    const auto so = TensorAccessor(o_args, o_addr, PAGE);
    for (uint32_t u = u0; u < u0 + nu; ++u) {
        cb_wait_front(16, 1);
        noc_async_write(get_read_ptr(16), so.get_noc_addr((u / 4) * 4 + (u % 4)), PAGE);
        noc_async_writes_flushed();
        cb_pop_front(16, 1);
    }
    noc_async_write_barrier();
}
