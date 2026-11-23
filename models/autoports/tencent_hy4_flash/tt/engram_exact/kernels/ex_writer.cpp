// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Engram exact decode tail, generic writer (RISCV_0): up to 2 outputs, each from its own CB, in the reader's page
// modes (0: pages v HT + c J + j; 2: page c; 3: none). ct: [J, HT, accessor args x 2]; rt: [c, per output: addr,
// page bytes, nvec, mode, cb]
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

template <typename A>
inline void write_output(const A& args, uint32_t base, uint32_t J, uint32_t HT, uint32_t c) {
    const uint32_t addr = get_arg_val<uint32_t>(base), page = get_arg_val<uint32_t>(base + 1);
    const uint32_t nvec = get_arg_val<uint32_t>(base + 2), mode = get_arg_val<uint32_t>(base + 3);
    const uint32_t cb = get_arg_val<uint32_t>(base + 4);
    if (mode == 3) {
        return;
    }
    const auto s = TensorAccessor(args, addr, page);
    if (mode == 2) {
        cb_wait_front(cb, 1);
        noc_async_write(get_read_ptr(cb), s.get_noc_addr(c), page);
        noc_async_write_barrier();
        cb_pop_front(cb, 1);
        return;
    }
    for (uint32_t v = 0; v < nvec; ++v) {
        for (uint32_t j = 0; j < J; ++j) {
            cb_wait_front(cb, 1);
            noc_async_write(get_read_ptr(cb), s.get_noc_addr(v * HT + c * J + j), page);
            noc_async_writes_flushed();
            cb_pop_front(cb, 1);
        }
    }
    noc_async_write_barrier();
}

void kernel_main() {
    constexpr uint32_t J = get_compile_time_arg_val(0);
    constexpr uint32_t HT = get_compile_time_arg_val(1);
    constexpr auto a0 = TensorAccessorArgs<2>();
    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
    const uint32_t c = get_arg_val<uint32_t>(0);
    write_output(a0, 1, J, HT, c);
    write_output(a1, 6, J, HT, c);
}
