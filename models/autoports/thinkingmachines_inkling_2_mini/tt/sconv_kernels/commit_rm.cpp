// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Inkling short conv, prefill commit: state rows := seq[n .. n + 2] of the row-major fp32 sequence [s0, s1, s2, x_0 ..]
// that a prefill chunk's conv built (the 3 inputs ending at real row n - 1). n is a RUNTIME argument: one program
// serves every chunk length (a slice at a host-known offset compiled one program per offset, and a program first
// compiled after the decode trace captures can have its kernels overwritten by the replays).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t seq_addr = get_arg_val<uint32_t>(0);
    const uint32_t st_addr = get_arg_val<uint32_t>(1);
    const uint32_t n = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(0);
    constexpr uint32_t ROWB = get_compile_time_arg_val(1);
    constexpr auto seq_args = TensorAccessorArgs<2>();
    constexpr auto st_args = TensorAccessorArgs<seq_args.next_compile_time_args_offset()>();
    const auto s_seq = TensorAccessor(seq_args, seq_addr, ROWB);
    const auto s_st = TensorAccessor(st_args, st_addr, ROWB);
    const uint32_t buf = get_write_ptr(cb_scr);
    for (uint32_t r = 0; r < 3; ++r) {
        noc_async_read(s_seq.get_noc_addr(n + r), buf + r * ROWB, ROWB);
    }
    noc_async_read_barrier();
    for (uint32_t r = 0; r < 3; ++r) {
        noc_async_write(buf + r * ROWB, s_st.get_noc_addr(r), ROWB);
    }
    noc_async_write_barrier();
}
