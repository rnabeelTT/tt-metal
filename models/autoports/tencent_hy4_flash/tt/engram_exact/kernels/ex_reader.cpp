// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Engram exact decode tail, generic reader (RISCV_1): up to 4 inputs, each into its own CB.
// ct: [J, HT, accessor args x 4]; rt: [c, then per input: addr, page bytes, nvec, mode, cb]
//   mode 0: pages v HT + c J + j (v < nvec, j < J); 1: pages 0 .. nvec - 1; 2: page c; 3: none.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

template <typename A>
inline void read_input(const A& args, uint32_t base, uint32_t J, uint32_t HT, uint32_t c) {
    const uint32_t addr = get_arg_val<uint32_t>(base), page = get_arg_val<uint32_t>(base + 1);
    const uint32_t nvec = get_arg_val<uint32_t>(base + 2), mode = get_arg_val<uint32_t>(base + 3);
    const uint32_t cb = get_arg_val<uint32_t>(base + 4);
    if (mode == 3) {
        return;
    }
    const auto s = TensorAccessor(args, addr, page);
    const uint32_t n = mode == 0 ? nvec * J : (mode == 1 ? nvec : 1);
    cb_reserve_back(cb, n);
    uint32_t w = get_write_ptr(cb);
    if (mode == 0) {
        for (uint32_t v = 0; v < nvec; ++v) {
            for (uint32_t j = 0; j < J; ++j) {
                noc_async_read(s.get_noc_addr(v * HT + c * J + j), w, page);
                w += page;
            }
        }
    } else if (mode == 1) {
        for (uint32_t v = 0; v < nvec; ++v) {
            noc_async_read(s.get_noc_addr(v), w, page);
            w += page;
        }
    } else {
        noc_async_read(s.get_noc_addr(c), w, page);
    }
    noc_async_read_barrier();
    cb_push_back(cb, n);
}

void kernel_main() {
    constexpr uint32_t J = get_compile_time_arg_val(0);
    constexpr uint32_t HT = get_compile_time_arg_val(1);
    constexpr auto a0 = TensorAccessorArgs<2>();
    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
    constexpr auto a2 = TensorAccessorArgs<a1.next_compile_time_args_offset()>();
    constexpr auto a3 = TensorAccessorArgs<a2.next_compile_time_args_offset()>();
    const uint32_t c = get_arg_val<uint32_t>(0);
    read_input(a0, 1, J, HT, c);
    read_input(a1, 6, J, HT, c);
    read_input(a2, 11, J, HT, c);
    read_input(a3, 16, J, HT, c);
}
